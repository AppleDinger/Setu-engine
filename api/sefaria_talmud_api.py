from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import time
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


LOGGER = logging.getLogger("sefaria_talmud_api")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

SEFARIA_TEXT_URL = "https://www.sefaria.org/api/v3/texts/{ref}"
DEFAULT_VERSION = "english"
DEFAULT_COMMENTARY = "0"
DEFAULT_CONTEXT = "0"
DEFAULT_SLEEP_SECONDS = 1.0
DEFAULT_TALMUD_EXPORT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "FullTalmud.txt"

DEFAULT_TRACTATES = [
    "Berakhot",
    "Shabbat",
    "Eruvin",
    "Pesachim",
    "Shekalim",
    "Yoma",
    "Sukkah",
    "Beitzah",
    "Rosh Hashanah",
    "Taanit",
    "Megillah",
    "Moed Katan",
    "Chagigah",
    "Yevamot",
    "Ketubot",
    "Nedarim",
    "Nazir",
    "Sotah",
    "Gittin",
    "Kiddushin",
    "Bava Kamma",
    "Bava Metzia",
    "Bava Batra",
    "Sanhedrin",
    "Makkot",
    "Shevuot",
    "Avodah Zarah",
    "Horayot",
    "Zevachim",
    "Menachot",
    "Chullin",
    "Bekhorot",
    "Arachin",
    "Temurah",
    "Keritot",
    "Meilah",
    "Tamid",
    "Middot",
    "Niddah",
]

HTML_TAG_RE = re.compile(r"<[^>]+>")
SUPERSCRIPT_TAG_RE = re.compile(r"<sup[^>]*>.*?</sup>", re.IGNORECASE | re.DOTALL)
BRACKETED_FOOTNOTE_RE = re.compile(r"\[[^\[\]]{1,40}\]")
PARENTHETICAL_CROSSREF_RE = re.compile(
    r"\(\s*(?:"
    r"(?:ibid\.|id\.|cf\.|see|supra|infra|loc\. cit\.)[^)]*|"
    r"(?:\d+[ab]?:?\d*(?:-\d+[ab]?:?\d*)?|[ivxlcdm]+(?:\s*[ab])?)"
    r")\s*\)",
    re.IGNORECASE,
)
MULTISPACE_RE = re.compile(r"\s+")


class TractateExportRequest(BaseModel):
    tractates: list[str] = Field(default_factory=list, description="List of Talmud tractates")
    all_tractates: bool = Field(default=False, description="Fetch all default Bavli tractates")
    output_format: str = Field(default="json", pattern="^(json|csv|txt)$")
    output_path: str | None = Field(default=None, description="Optional output path")
    version: str = Field(default=DEFAULT_VERSION, description="Sefaria version selector")
    sleep_seconds: float = Field(default=DEFAULT_SLEEP_SECONDS, ge=0.0)


def create_session() -> requests.Session:
    """Create a retrying requests session for polite API access."""

    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Setu-engine/1.0 (+https://www.sefaria.org)"})
    return session


def normalize_tractate_name(name: str) -> str:
    """Normalize user input into the common Sefaria tractate naming style."""

    return " ".join(part.capitalize() if part else part for part in name.strip().split())


def to_sefaria_api_ref(ref: str) -> str:
    """Convert a Sefaria ref like 'Berakhot 2a' to a URL path ref like 'Berakhot.2a'."""

    ref = ref.strip()
    if "." in ref:
        return ref

    parts = ref.rsplit(" ", 1)
    if len(parts) == 2:
        return f"{parts[0]}.{parts[1]}"

    return ref.replace(" ", ".")


def build_text_url(ref: str) -> str:
    return SEFARIA_TEXT_URL.format(ref=quote(to_sefaria_api_ref(ref), safe="."))


def flatten_text(value: Any) -> list[str]:
    """Flatten nested lists of text fragments into a simple list of strings."""

    flattened: list[str] = []

    if value is None:
        return flattened

    if isinstance(value, str):
        if value.strip():
            flattened.append(value)
        return flattened

    if isinstance(value, Iterable):
        for item in value:
            flattened.extend(flatten_text(item))

    return flattened


def clean_talmud_text(text: str) -> str:
    """Remove HTML, footnote markers, and citation noise while preserving prose."""

    text = html.unescape(text)
    text = SUPERSCRIPT_TAG_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = BRACKETED_FOOTNOTE_RE.sub(" ", text)
    text = PARENTHETICAL_CROSSREF_RE.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def clean_text_block(text_fragments: Any) -> str:
    """Convert a Sefaria text payload into a single cleaned string."""

    cleaned_fragments = [clean_talmud_text(fragment) for fragment in flatten_text(text_fragments)]
    cleaned_fragments = [fragment for fragment in cleaned_fragments if fragment]
    return " ".join(cleaned_fragments)


def fetch_text_page(session: requests.Session, ref: str, version: str = DEFAULT_VERSION, timeout: int = 30) -> dict[str, Any]:
    """Fetch one daf from the Sefaria API."""

    response = session.get(
        build_text_url(ref),
        params={
            "version": version,
            "commentary": DEFAULT_COMMENTARY,
            "context": DEFAULT_CONTEXT,
        },
        timeout=timeout,
    )

    if response.status_code == 404:
        raise FileNotFoundError(f"Sefaria returned 404 for ref {ref}")

    if response.status_code >= 400:
        raise requests.HTTPError(
            f"Sefaria request failed for {ref} with status {response.status_code}: {response.text[:200]}",
            response=response,
        )

    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected payload type for {ref}: {type(data)!r}")

    return data


def extract_english_text(data: dict[str, Any]) -> tuple[str, str | None]:
    """Pick the English version from a Sefaria response and clean it."""

    versions = data.get("versions") or []
    if not versions:
        raise ValueError("No versions returned by Sefaria")

    selected_version = None
    for version in versions:
        if version.get("language") == "en" or version.get("actualLanguage") == "en":
            selected_version = version
            break

    if selected_version is None:
        raise ValueError("Sefaria did not return an English version")

    cleaned_text = clean_text_block(selected_version.get("text"))
    return cleaned_text, selected_version.get("versionTitle")


def iterate_tractate(tractate: str, session: requests.Session, version: str, sleep_seconds: float) -> list[dict[str, str]]:
    """Iterate through every daf in a tractate using Sefaria's next-ref metadata."""

    normalized = normalize_tractate_name(tractate)
    current_ref: str | None = f"{normalized} 2a"
    rows: list[dict[str, str]] = []

    while current_ref:
        data = fetch_text_page(session, current_ref, version=version)
        cleaned_text, _version_title = extract_english_text(data)

        if cleaned_text:
            rows.append(
                {
                    "tractate": normalized,
                    "daf": str(data.get("sectionRef", current_ref)).split()[-1],
                    "text": cleaned_text,
                }
            )

        next_ref = data.get("next")
        LOGGER.info("Fetched %s -> next=%s", current_ref, next_ref)
        current_ref = next_ref if isinstance(next_ref, str) and next_ref.strip() else None

        if current_ref:
            time.sleep(sleep_seconds)

    return rows


def export_rows(rows: list[dict[str, str]], output_format: str, output_path: str | None) -> Path:
    """Write tractate rows to disk as JSON or CSV."""

    if output_path is None:
        output_dir = Path.cwd() / "api_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"sefaria_talmud_cleaned.{output_format}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "json":
        destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    elif output_format == "csv":
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["tractate", "daf", "text"])
            writer.writeheader()
            writer.writerows(rows)
    elif output_format == "txt":
        if destination.suffix.lower() == ".txt":
            grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped_rows[row["tractate"]].append(row)

            combined_lines: list[str] = []
            for tractate in sorted(grouped_rows):
                combined_lines.append(f"# Tractate: {tractate}")
                for row in grouped_rows[tractate]:
                    combined_lines.append(f"## Daf: {row['daf']}")
                    combined_lines.append(row["text"])
                    combined_lines.append("")
                combined_lines.append("")

            destination.write_text("\n".join(combined_lines).strip() + "\n", encoding="utf-8")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                grouped_rows[row["tractate"]].append(row)

            for tractate, tractate_rows in grouped_rows.items():
                safe_name = re.sub(r"[^A-Za-z0-9]+", "_", tractate).strip("_").lower()
                tractate_path = destination / f"{safe_name}.txt"
                lines: list[str] = [f"# Tractate: {tractate}", ""]
                for row in tractate_rows:
                    lines.append(f"## Daf: {row['daf']}")
                    lines.append(row["text"])
                    lines.append("")
                tractate_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    return destination


def extract_tractates(
    tractates: list[str] | None = None,
    all_tractates: bool = False,
    output_format: str = "json",
    output_path: str | None = None,
    version: str = DEFAULT_VERSION,
    sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
) -> dict[str, Any]:
    """Fetch, clean, and export Talmud tractates from Sefaria."""

    selected_tractates = DEFAULT_TRACTATES if all_tractates else (tractates or [])
    if not selected_tractates:
        raise ValueError("Provide at least one tractate or set all_tractates=True")

    session = create_session()
    all_rows: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for tractate in selected_tractates:
        try:
            rows = iterate_tractate(tractate, session=session, version=version, sleep_seconds=sleep_seconds)
            all_rows.extend(rows)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %s: %s", tractate, exc)
            errors.append({"tractate": tractate, "error": str(exc)})
        except (requests.Timeout, requests.ConnectionError) as exc:
            LOGGER.warning("Skipping %s after connection issue: %s", tractate, exc)
            errors.append({"tractate": tractate, "error": str(exc)})
        except Exception as exc:  # pragma: no cover - final safeguard for API variability
            LOGGER.exception("Unexpected failure while processing %s", tractate)
            errors.append({"tractate": tractate, "error": str(exc)})

    exported_path = export_rows(all_rows, output_format=output_format, output_path=output_path)

    return {
        "tractates": selected_tractates,
        "row_count": len(all_rows),
        "exported_path": str(exported_path),
        "errors": errors,
        "output_format": output_format,
    }


app = FastAPI(title="Sefaria Talmud Extractor", version="1.0.0")


@app.get("/tractates")
def list_supported_tractates() -> dict[str, list[str]]:
    return {"tractates": DEFAULT_TRACTATES}


@app.post("/extract")
def extract_endpoint(request: TractateExportRequest) -> dict[str, Any]:
    try:
        return extract_tractates(
            tractates=request.tractates,
            all_tractates=request.all_tractates,
            output_format=request.output_format,
            output_path=request.output_path,
            version=request.version,
            sleep_seconds=request.sleep_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch and clean Talmud tractates from Sefaria.")
    parser.add_argument(
        "tractates",
        nargs="*",
        help="Tractates to fetch, or use 'talmud' to fetch the default full Bavli set",
    )
    parser.add_argument("--all", action="store_true", help="Fetch all default Bavli tractates")
    parser.add_argument("--format", choices=["json", "csv", "txt"], default="json", help="Output file format")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--version", default=DEFAULT_VERSION, help="Sefaria version selector")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP_SECONDS, help="Sleep seconds between daf requests")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    requested_tractates = list(args.tractates)
    talmud_shortcut = len(requested_tractates) == 1 and requested_tractates[0].lower() == "talmud"
    selected_format = "txt" if talmud_shortcut and args.format == "json" else args.format
    selected_output_path = args.output

    if talmud_shortcut:
        requested_tractates = []
        if selected_output_path is None:
            selected_output_path = str(DEFAULT_TALMUD_EXPORT_PATH.with_suffix(f".{selected_format}"))

    result = extract_tractates(
        tractates=requested_tractates,
        all_tractates=args.all or talmud_shortcut,
        output_format=selected_format,
        output_path=selected_output_path,
        version=args.version,
        sleep_seconds=args.sleep,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()