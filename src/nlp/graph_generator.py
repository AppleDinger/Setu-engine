import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy

class EntityExtractionEngine:
    def __init__(self, mapping_url: str):
        # Load the base model
        self.nlp = spacy.load("en_core_web_md")
        self.alias_lookup = {}
        self.load_and_build_registry(mapping_url)
        self.inject_entity_ruler()

    def load_and_build_registry(self, url: str):
        """Loads the local registry file and builds a fast flat lookup map."""
        registry_path = Path(url)
        print(f"Syncing Identity Registry from: {registry_path}")
        with open(registry_path, "r", encoding="utf-8") as handle:
            registry_data = json.load(handle)
        
        self.rules = []
        for entity in registry_data.get("entities", []):
            canonical = entity["canonical_name"]
            
            # Map canonical name to itself (lowercase key)
            self.alias_lookup[canonical.lower()] = canonical
            self.rules.append({"label": "ANCIENT_HERO", "pattern": canonical})
            
            # Map alternative variations/aliases
            for alias in entity.get("aliases", []):
                self.alias_lookup[alias.lower()] = canonical
                self.rules.append({"label": "ANCIENT_HERO", "pattern": alias})

    def inject_entity_ruler(self):
        """Injects deterministic patterns with overwrite capabilities."""
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")
        
        # Configure ruler to overwrite existing statistical NER components
        config = {"overwrite_ents": True}
        ruler = self.nlp.add_pipe("entity_ruler", before="ner", config=config)
        ruler.add_patterns(self.rules)
        print("Forced EntityRuler successfully injected into spaCy pipeline.")

    def extract_canonical_entities(self, word_window: list[str]) -> set[str]:
        """Extracts unique verified canonical identities from a text window."""
        window_text = " ".join(word_window)
        doc = self.nlp(window_text)
        found_entities = set()
        
        # DEBUG CHECK: Uncomment this line if you need to see what spaCy tags on screen
        # print(f"DEBUG WINDOW TOKENS: {[(ent.text, ent.label_) for ent in doc.ents]}")
        
        for ent in doc.ents:
            # Match our custom tag, or match standard tags if they exist in our registry lookup
            if ent.label_ in ["ANCIENT_HERO", "PERSON"]:
                ent_clean = ent.text.strip()
                lookup_key = ent_clean.lower()

                if lookup_key in self.alias_lookup:
                    found_entities.add(self.alias_lookup[lookup_key])
                elif ent.label_ == "PERSON" and ent_clean:
                    # Preserve named entities that are not present in the registry.
                    found_entities.add(ent_clean)
        return found_entities

def generate_edge_list(text: str, engine: EntityExtractionEngine, window_size=100, stride=25) -> pd.DataFrame:
    """Iterates over text windows, extracts entities, and tabulates weighted edges."""
    from processing.clean_text import sliding_window_words
    
    edge_weights = defaultdict(int)
    windows = list(sliding_window_words(text, window_size=window_size, stride=stride))
    
    print(f"Total windows to process: {len(windows)}")
    
    for idx, window in enumerate(windows):
        entities = list(engine.extract_canonical_entities(window))
        
        # If at least two distinct canonical entities appear together, map the edge
        if len(entities) > 1:
            entities.sort()
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    source = entities[i]
                    target = entities[j]
                    edge_weights[(source, target)] += 1

    edge_data = []
    for (source, target), weight in edge_weights.items():
        edge_data.append({"Source": source, "Target": target, "Weight": weight})
        
    if not edge_data:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
        
    return pd.DataFrame(edge_data)