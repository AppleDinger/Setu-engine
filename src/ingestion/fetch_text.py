"""Batch fetch primary texts from Project Gutenberg.

This ingestion script is designed to be source-agnostic: add a Gutenberg ID + filename,
and the pipeline can ingest another literary tradition without changing core logic.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import requests
import re


# Curated corpus across Indo-Aryan and Levantine/Semitic traditions.
TEXT_SOURCES: list[dict[str, str | int]] = [
    # INDO-ARYAN CLUSTER (SANSKRIT TRADITIONS)
    {"id": 15474, "name": "Mahabharata, Vol. 1", "filename": "mahabharata_vol1_raw.txt", "cluster": "indo_aryan"},
    {"id": 15475, "name": "Mahabharata, Vol. 2", "filename": "mahabharata_vol2_raw.txt", "cluster": "indo_aryan"},
    {"id": 15476, "name": "Mahabharata, Vol. 3", "filename": "mahabharata_vol3_raw.txt", "cluster": "indo_aryan"},
    {"id": 15477, "name": "Mahabharata, Vol. 4", "filename": "mahabharata_vol4_raw.txt", "cluster": "indo_aryan"},
    {"id": 24869, "name": "Ramayana (Valmiki)", "filename": "ramayana_raw.txt", "cluster": "indo_aryan"},
    {"id": 2388, "name": "The Bhagavad-Gita", "filename": "bhagavad_gita_raw.txt", "cluster": "indo_aryan"},
    {"id": 66208, "name": "Vishnupuranam", "filename": "vishnupuranam_raw.txt", "cluster": "indo_aryan"},

    # LEVANTINE & SEMITIC CLUSTER (ANCIENT NEAR EAST)
    {"id": 10, "name": "The King James Bible", "filename": "king_james_bible_raw.txt", "cluster": "levantine_semitic"},
    {"id": 2848, "name": "Antiquities of the Jews", "filename": "josephus_antiquities_raw.txt", "cluster": "levantine_semitic"},
    {"id": 2850, "name": "The Wars of the Jews", "filename": "josephus_wars_raw.txt", "cluster": "levantine_semitic"},
    {"id": 3434, "name": "The Koran (Al-Qur'an)", "filename": "quran_raw.txt", "cluster": "levantine_semitic"},
    {"id": 1493, "name": "Legends of the Jews, Vol. 1", "filename": "legends_of_the_jews_v1_raw.txt", "cluster": "levantine_semitic"},
    {"id": 1494, "name": "Legends of the Jews, Vol. 2", "filename": "legends_of_the_jews_v2_raw.txt", "cluster": "levantine_semitic"},
    {"id": 2881, "name": "Legends of the Jews, Vol. 3", "filename": "legends_of_the_jews_v3_raw.txt", "cluster": "levantine_semitic"},
    {"id": 2882, "name": "Legends of the Jews, Vol. 4", "filename": "legends_of_the_jews_v4_raw.txt", "cluster": "levantine_semitic"},
    {"id": 7793, "name": "The Book of Enoch", "filename": "book_of_enoch_raw.txt", "cluster": "levantine_semitic"},
]


def candidate_urls(gutenberg_id: int) -> list[str]:
    """Return URL fallbacks for Gutenberg plain text files."""
    gid = str(gutenberg_id)
    return [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
        f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt",
        f"https://www.gutenberg.org/ebooks/{gid}.txt.utf-8",
    ]


def fetch_text(gutenberg_id: int, timeout: int = 30) -> tuple[str, str]:
    """Fetch text content for a Gutenberg ID, returning (content, source_url)."""
    headers = {
        "User-Agent": "SetuEngineIngestion/1.0 (research; reproducible corpus fetch)",
    }

    last_error: Exception | None = None
    for url in candidate_urls(gutenberg_id):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200 and response.text.strip():
                if response.encoding is None:
                    response.encoding = response.apparent_encoding or "utf-8"
                return response.text, url
        except requests.RequestException as exc:
            last_error = exc

    if last_error is not None:
        raise RuntimeError(
            f"Failed to fetch Gutenberg ID {gutenberg_id}. Last error: {last_error}"
        ) from last_error

    raise RuntimeError(f"No valid text endpoint returned content for ID {gutenberg_id}.")


def default_output_dir() -> Path:
    """Resolve output path.

    Priority:
    1) DATA_REPO_RAW_DIR env var (for external data repo /raw folder)
    2) <engine_repo>/data/raw
    """
    env_path = os.getenv("DATA_REPO_RAW_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "raw"


def download_all(
    sources: Iterable[dict[str, str | int]],
    output_dir: Path,
    overwrite: bool = False,
    timeout: int = 30,
    throttle_seconds: float = 0.2,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    succeeded = 0
    skipped = 0
    failed = 0

    for source in sources:
        total += 1
        gid = int(source["id"])
        title = str(source["name"])
        filename = str(source["filename"])
        destination = output_dir / filename

        if destination.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] {gid} {title} -> {destination}")
            continue

        try:
            text, source_url = fetch_text(gid, timeout=timeout)
            destination.write_text(text, encoding="utf-8")
            succeeded += 1
            print(f"[OK]   {gid} {title} -> {destination} (from {source_url})")
        except Exception as exc:  # noqa: BLE001 - collect all failures in one run
            failed += 1
            print(f"[FAIL] {gid} {title}: {exc}")

        if throttle_seconds > 0:
            time.sleep(throttle_seconds)

    print("\nIngestion complete.")
    print(f"Total: {total} | Success: {succeeded} | Skipped: {skipped} | Failed: {failed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch curated Project Gutenberg texts into data/raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Destination folder for downloaded .txt files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and overwrite files that already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP timeout in seconds per request.",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.2,
        help="Delay in seconds between downloads.",
    )
    parser.add_argument(
        "--rename-existing",
        action="store_true",
        help="Attempt to rename existing files in the output dir to the canonical filenames.",
    )
    return parser.parse_args()


def rename_existing_files(output_dir: Path, sources: Iterable[dict[str, str | int]]) -> None:
    """Heuristically rename files in `output_dir` to match canonical filenames in `sources`.

    The function looks for files whose current filename contains tokens from the
    source `name` and renames the first match to the canonical `filename`.
    This is best-effort and runs only if the user opts in via `--rename-existing`.
    """
    if not output_dir.exists():
        return

    files = list(output_dir.iterdir())
    for source in sources:
        target = output_dir / str(source["filename"])
        if target.exists():
            continue

        name = str(source["name"]).lower()
        tokens = [t for t in re.findall(r"\w+", name) if len(t) > 3]
        if not tokens:
            continue

        for f in files:
            if not f.is_file():
                continue
            if f.name == target.name:
                break
            fname = f.name.lower()
            if any(tok in fname for tok in tokens):
                try:
                    f.rename(target)
                    print(f"[RENAME] {f.name} -> {target.name}")
                    break
                except Exception as exc:
                    print(f"[RENAME-ERR] {f.name} -> {target.name}: {exc}")
                    break


def main() -> None:
    args = parse_args()
    print(
        "Running source-agnostic corpus ingestion: add a Gutenberg ID to TEXT_SOURCES "
        "to scale to new traditions."
    )
    print(f"Output directory: {args.output_dir}\n")
    if getattr(args, "rename_existing", False):
        print("Attempting to rename existing files to canonical filenames...")
        rename_existing_files(args.output_dir, TEXT_SOURCES)

    download_all(
        sources=TEXT_SOURCES,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
        throttle_seconds=args.throttle,
    )


if __name__ == "__main__":
    main()
