import json
from pathlib import Path
import spacy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_ROOT / "alias mapping" / "mapping.json"

def load_local_registry(path):
    """Fetches the identity registry JSON file from the local project root."""
    print(f"Fetching registry from local storage: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as e:
        raise RuntimeError(f"Local registry error: {e}")

def build_alias_map(registry_data):
    """
    Flattens the registry structure into a direct lookup dictionary.
    Keys are lowercased aliases, values are the Canonical Names.
    """
    alias_map = {}
    for entity in registry_data.get("entities", []):
        canonical = entity["canonical_name"]
        
        # A character's canonical name is also an alias for themselves
        alias_map[canonical.lower()] = canonical
        
        # Add all other listed aliases
        for alias in entity.get("aliases", []):
            alias_map[alias.lower()] = canonical
            
    return alias_map

def test_resolution():
    # 1. Initialize spaCy model
    print("Loading NLP model...")
    nlp = spacy.load("en_core_web_md")
    
    try:
        registry_data = load_local_registry(REGISTRY_PATH)
        alias_lookup = build_alias_map(registry_data)
        print("Registry successfully loaded from local mapping.json and mapped.")
    except Exception as e:
        print(f"Error loading registry: {e}")
        print("\nFalling back to an in-memory mock registry for testing...")
        alias_lookup = {
            "dhananjaya": "Arjuna",
            "partha": "Arjuna",
            "elohim": "Yahweh",
            "krishna": "Krishna"
        }

    # 3. Sample input text
    sample_text = "Dhananjaya rode into battle alongside Krishna."
    print(f"\nProcessing Text: '{sample_text}'")
    
    # 4. Process text with spaCy
    doc = nlp(sample_text)
    
    # 5. Entity/Token Resolution Loop
    print("\n--- Identity Resolution Results ---")
    for token in doc:
        token_text_lower = token.text.lower()
        
        if token_text_lower in alias_lookup:
            canonical_identity = alias_lookup[token_text_lower]
            print(f"Found Alias: '{token.text}' -> Resolves to Canonical Identity: '{canonical_identity}'")

if __name__ == "__main__":
    test_resolution()