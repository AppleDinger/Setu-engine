import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from spacy.language import Language

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

# For GPU runs, install the matching spaCy CUDA build first:
# pip install "spacy[cuda12x]"  # or the CUDA extra that matches your environment.
SPACY_GPU_PREFERRED = spacy.prefer_gpu()
SPACY_BATCH_SIZE = 2000 if SPACY_GPU_PREFERRED else 256
SPACY_RUNTIME_MODE = "GPU" if SPACY_GPU_PREFERRED else "CPU"
USE_STATISTICAL_NER = False


def get_torch_cuda_status() -> str:
    """Return a short runtime summary for PyTorch CUDA support."""

    if torch is None:
        return "PyTorch not installed"

    if not torch.cuda.is_available():
        return "PyTorch CUDA unavailable"

    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "unknown"
    return f"PyTorch CUDA available ({device_count} device(s), primary={device_name})"


def get_pipe_disable_components(nlp: Language) -> list[str]:
    """Return non-essential components that can be skipped during entity extraction."""

    disabled = [name for name in ["parser", "attribute_ruler", "lemmatizer"] if name in nlp.pipe_names]

    # tok2vec is required by the statistical ner component, so only keep it when ner is enabled.
    if not USE_STATISTICAL_NER and "tok2vec" in nlp.pipe_names:
        disabled.append("tok2vec")
    elif "ner" not in nlp.pipe_names and "tok2vec" in nlp.pipe_names:
        disabled.append("tok2vec")

    return disabled

class EntityExtractionEngine:
    def __init__(self, mapping_url: str):
        # Load the smallest pipeline that still supports the configured extraction mode.
        if USE_STATISTICAL_NER:
            self.nlp = spacy.load(
                "en_core_web_md",
                disable=["parser", "attribute_ruler", "lemmatizer"],
            )
        else:
            self.nlp = spacy.blank("en")
        self.alias_lookup = {}
        self.load_and_build_registry(mapping_url)
        self.inject_entity_ruler()
        self.pipeline_ready = True
        print(
            f"spaCy runtime mode: {SPACY_RUNTIME_MODE} "
            f"(batch_size={SPACY_BATCH_SIZE}, ner_enabled={USE_STATISTICAL_NER})"
        )
        print(f"CUDA runtime status: {get_torch_cuda_status()}")
        print(f"spaCy pipeline status: {'READY' if self.pipeline_ready else 'NOT READY'}")

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
        if USE_STATISTICAL_NER and "ner" in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner", config=config)
        else:
            ruler = self.nlp.add_pipe("entity_ruler", config=config)
        ruler.add_patterns(self.rules)
        print("Forced EntityRuler successfully injected into spaCy pipeline.")

    def extract_canonical_entities(self, word_window: list[str]) -> set[str]:
        """Extracts unique verified canonical identities from a text window."""
        window_text = " ".join(word_window)
        doc = self.nlp(window_text)
        return self.extract_canonical_entities_from_doc(doc)

    def extract_canonical_entities_from_doc(self, doc) -> set[str]:
        """Extract canonical identities from a preprocessed spaCy Doc."""
        found_entities = set()

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
    from src.processing.clean_text import sliding_window_words
    
    edge_weights = defaultdict(int)
    window_texts = (" ".join(window) for window in sliding_window_words(text, window_size=window_size, stride=stride))
    disabled_components = get_pipe_disable_components(engine.nlp)

    print(f"Pipeline working status: {'OK' if getattr(engine, 'pipeline_ready', False) else 'NOT OK'}")
    print(
        f"Streaming windows through spaCy.pipe with batch_size={SPACY_BATCH_SIZE} "
        f"and disabled_components={disabled_components}"
    )
    print(f"GPU detected: {'YES' if SPACY_GPU_PREFERRED else 'NO, using CPU'}")
    print(f"Fast pipeline mode: {'NO' if USE_STATISTICAL_NER else 'YES - entity ruler only'}")

    processed_windows = 0

    for idx, doc in enumerate(
        engine.nlp.pipe(window_texts, batch_size=SPACY_BATCH_SIZE, disable=disabled_components),
        start=1,
    ):
        processed_windows = idx
        entities = list(engine.extract_canonical_entities_from_doc(doc))
        
        # If at least two distinct canonical entities appear together, map the edge
        if len(entities) > 1:
            entities.sort()
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    source = entities[i]
                    target = entities[j]
                    edge_weights[(source, target)] += 1

        if idx % 5000 == 0:
            print(f"Processed {idx} windows...")

    print(f"Pipeline completed: processed {processed_windows} windows")

    edge_data = []
    for (source, target), weight in edge_weights.items():
        edge_data.append({"Source": source, "Target": target, "Weight": weight})
        
    if not edge_data:
        return pd.DataFrame(columns=["Source", "Target", "Weight"])
        
    return pd.DataFrame(edge_data)