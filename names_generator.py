import sys
from pathlib import Path
from collections import Counter
from typing import Iterator

import requests
import spacy

# Ensure the src folder is in the python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from processing.clean_text import clean_text

REGISTRY_URLS = [
    "https://raw.githubusercontent.com/AppleDinger/Setu-dataset/main/registry/mapping.json",
    "https://raw.githubusercontent.com/AppleDinger/Setu-dataset/main/mapping.json",
]

FALLBACK_ALIAS_LOOKUP = {
    "dhananjaya": "Arjuna",
    "partha": "Arjuna",
    "savyasachi": "Arjuna",
    "son of kunti": "Arjuna",
    "krishna": "Krishna",
    "govinda": "Krishna",
    "madhava": "Krishna",
    "keshava": "Krishna",
}


def build_alias_map(registry_data: dict) -> dict[str, str]:
    """Flatten the registry into a lowercased alias-to-canonical lookup."""
    alias_map = {}
    for entity in registry_data.get("entities", []):
        canonical = entity.get("canonical_name")
        if not canonical:
            continue

        alias_map[canonical.lower()] = canonical
        for alias in entity.get("aliases", []):
            alias_map[alias.lower()] = canonical

    return alias_map


def load_alias_lookup() -> dict[str, str]:
    """Load registry aliases from GitHub, then fall back to a local minimal map."""
    for url in REGISTRY_URLS:
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            alias_lookup = build_alias_map(response.json())
            if alias_lookup:
                print(f"Loaded identity registry from: {url}")
                return alias_lookup
        except Exception:
            continue

    print("Using built-in alias fallback for name aggregation.")
    return FALLBACK_ALIAS_LOOKUP.copy()


def iter_text_chunks(file_path: Path, chunk_char_limit: int = 25_000) -> Iterator[str]:
    """Yield raw text chunks without cutting lines in half."""
    buffer = []
    buffer_size = 0
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            buffer.append(line)
            buffer_size += len(line)
            if buffer_size >= chunk_char_limit:
                yield "".join(buffer)
                buffer = []
                buffer_size = 0

    if buffer:
        yield "".join(buffer)


def run_scout_pipeline():
    print("Initializing identity scouting run...")
    
    # FIX: Force spaCy to configure its internal array engine to use your RTX 4060
    try:
        spacy.require_gpu()
        print("🚀 Success: spaCy GPU acceleration activated via CUDA 13!")
    except Exception as e:
        print(f"⚠️ GPU Initialization failed: {e}")
        print("Falling back to CPU allocation mode.")

    # Load the NLP pipeline framework into memory
    nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    alias_lookup = load_alias_lookup()
    
    # Use the local raw corpus files that already exist in this repository.
    raw_candidates = [
        PROJECT_ROOT / "data" / "raw" / "mahabharata_raw.txt",
    ]
    raw_candidates.extend(sorted((PROJECT_ROOT / "data" / "raw").glob("mahabharata*.txt")))

    raw_master_path = next((path for path in raw_candidates if path.exists()), None)

    if raw_master_path is None:
        print("❌ Error: No Mahabharata raw file was found in data/raw.")
        return

    print(f"Reading and normalizing text corpus from {raw_master_path.name}...")
    raw_counts = Counter()
    canonical_counts = Counter()

    print("Running spaCy NER over the full corpus in chunks...")
    cleaned_chunks = (clean_text(chunk) for chunk in iter_text_chunks(raw_master_path))
    
    # Using smaller chunk sizes with batch pipelining maxes out your 4060's processing efficiency
    for doc in nlp.pipe(cleaned_chunks, batch_size=32):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name = ent.text.strip()
            if len(name) <= 1:
                continue

            raw_counts[name] += 1
            canonical_name = alias_lookup.get(name.lower(), name)
            canonical_counts[canonical_name] += 1

    print("\n--- Top 30 Canonical Identities Found by AI ---")
    print("Alias-resolved counts across the full Mahabharata corpus")
    print("--------------------------------------------------")
    for name, count in canonical_counts.most_common(30):
        print(f"  Character: {name:<20} | Mentions: {count}")

    print("\n--- Top 30 Raw Person Entities Found by AI ---")
    print("Use these values to expand your public mapping.json registry:")
    print("--------------------------------------------------")
    for name, count in raw_counts.most_common(30):
        print(f"  Character: {name:<20} | Mentions: {count}")


if __name__ == "__main__":
    run_scout_pipeline()