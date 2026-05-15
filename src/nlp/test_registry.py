import json
import spacy
import requests

def load_remote_registry(url):
    """Fetches the identity registry JSON file directly from GitHub."""
    print(f"Fetching registry from remote storage: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()  # Throws an exception for 4xx or 5xx errors
        return response.json()
    except Exception as e:
        raise RuntimeError(f"Network error fetching registry: {e}")

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
    
    # 2. Public Raw GitHub URL for your dataset mapping file
    raw_github_url = "https://raw.githubusercontent.com/AppleDinger/Setu-dataset/main/registry/mapping.json"
    
    try:
        registry_data = load_remote_registry(raw_github_url)
        alias_lookup = build_alias_map(registry_data)
        print("Registry successfully loaded from GitHub and mapped.")
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