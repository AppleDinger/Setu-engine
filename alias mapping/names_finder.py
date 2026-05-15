import sys
import json
from pathlib import Path
from collections import Counter
from typing import Iterator

import spacy

# Ensure the repo src folder is in the python path
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RAW_CANDIDATES = [
    REPO_ROOT / "data" / "raw" / "book_of_enoch_raw.txt",
]

from processing.clean_text import clean_text

REGISTRY_URLS = [
    REPO_ROOT / "alias mapping" / "mapping.json",
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
    "kesava": "Krishna",
}

TRANSLITERATION_NORMALIZATION = {
    "kesava": "keshava",
}

HONORIFIC_PREFIXES = {
    "r", "r.", "rabbi", "rav", "reb", "rabban", "rabbenu",
}

FALSE_PERSON_TERMS = {
    "anonymous",
    "chapter",
    "editor",
    "footnote",
    "footnotes",
    "gutenberg",
    "lord",
    "notes",
    "project gutenberg",
    "translator",
    "the lord",
    "thee",
    "thou",
    "thy",
    "thine",
    "ye",
    "art",
    "hast",
    "hath",
    "wilt",
    "shalt",
}

# Additional corpus-specific stopwords common in Talmudic texts
FALSE_PERSON_TERMS.update({
    "sabbath",
    "boraitha",
    "baraita",
    "baraitah",
    "torah",
    "gemara",
    "law",
    "tana",
    "tanaim",
    "said",
    "said ",
})


def build_alias_map(registry_data: dict) -> dict[str, str]:
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
    registry_path = REGISTRY_URLS[0]
    try:
        with open(registry_path, "r", encoding="utf-8") as handle:
            alias_lookup = build_alias_map(json.load(handle))
            if alias_lookup:
                print(f"Loaded identity registry from: {registry_path}")
                return alias_lookup
    except Exception:
        pass

    print("Using built-in alias fallback for name aggregation.")
    return FALLBACK_ALIAS_LOOKUP.copy()


def normalize_person_name_for_lookup(name: str) -> str:
    if not name:
        return ""

    s = name.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    parts = s.split()
    if parts:
        first = parts[0].rstrip('.')
        if first.lower() in HONORIFIC_PREFIXES:
            parts = parts[1:]

    if parts and parts[0].lower() in {"said", "says", "said:"}:
        parts = parts[1:]

    key = " ".join(parts).strip().strip('.,:;()[]"')
    key = TRANSLITERATION_NORMALIZATION.get(key.lower(), key.lower())
    return key


def should_count_person_entity(name: str, alias_lookup: dict[str, str]) -> bool:
    cleaned_name = name.strip()
    if len(cleaned_name) <= 1:
        return False

    lookup_key = normalize_person_name_for_lookup(cleaned_name)
    if not lookup_key:
        return False

    if lookup_key in FALSE_PERSON_TERMS:
        return False

    return True


def iter_text_chunks(file_path: Path, chunk_char_limit: int = 25_000) -> Iterator[str]:
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
    try:
        spacy.require_gpu()
        print("🚀 Success: spaCy GPU acceleration activated via CUDA 13!")
    except Exception as e:
        print(f"⚠️ GPU Initialization failed: {e}")
        print("Falling back to CPU allocation mode.")

    nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "lemmatizer", "attribute_ruler"])
    alias_lookup = load_alias_lookup()

    raw_candidates = list(RAW_CANDIDATES)

    raw_master_path = next((path for path in raw_candidates if path.exists()), None)

    if raw_master_path is None:
        print("❌ Error: No raw file was found for the selected candidates in data/raw.")
        return

    book_label = raw_master_path.stem.replace("_raw", "").replace("_", " ").title()

    print(f"Reading and normalizing text corpus from {raw_master_path.name}...")
    raw_counts = Counter()
    canonical_counts = Counter()

    print("Running spaCy NER over the full corpus in chunks...")
    cleaned_chunks = (clean_text(chunk) for chunk in iter_text_chunks(raw_master_path))

    for doc in nlp.pipe(cleaned_chunks, batch_size=32):
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name = ent.text.strip()
            if not should_count_person_entity(name, alias_lookup):
                continue

            raw_counts[name] += 1

            lookup_key = normalize_person_name_for_lookup(name)
            canonical_name = alias_lookup.get(lookup_key, name)
            canonical_counts[canonical_name] += 1

    print("\n--- Top 50 Canonical Identities Found by AI ---")
    print(f"Alias-resolved counts across the full {book_label} corpus")
    print("--------------------------------------------------")
    for name, count in canonical_counts.most_common(50):
        print(f"  Character: {name:<20} | Mentions: {count}")

    print("\n--- Top 50 Raw Person Entities Found by AI ---")
    print("Use these values to expand your local mapping.json registry:")
    print("--------------------------------------------------")
    for name, count in raw_counts.most_common(50):
        print(f"  Character: {name:<20} | Mentions: {count}")


if __name__ == "__main__":
    # Interactive selection: prompt for a book name or filename, then run pipeline
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw"

    book = input("Enter book to extract names from: ").strip()
    if not book:
        run_scout_pipeline()
    else:
        candidate_path = Path(book)
        if candidate_path.exists():
            RAW_CANDIDATES = [candidate_path]
        else:
            # find files containing the book string
            matches = sorted(raw_dir.glob(f"*{book}*.txt"))
            if not matches:
                # try appending _raw pattern
                matches = sorted(raw_dir.glob(f"*{book}_raw*.txt"))

            if not matches:
                print(f"❌ No raw files matching '{book}' found in {raw_dir}.")
                raise SystemExit(1)

            if len(matches) == 1:
                RAW_CANDIDATES = [matches[0]]
            else:
                print("Multiple matching raw files found:")
                for i, p in enumerate(matches, start=1):
                    print(f"  {i}. {p.name}")
                choice = input("Choose file number (default 1): ").strip()
                try:
                    idx = int(choice) - 1
                    RAW_CANDIDATES = [matches[idx]]
                except Exception:
                    RAW_CANDIDATES = [matches[0]]

        run_scout_pipeline()
