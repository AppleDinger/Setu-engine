"""Utilities for cleaning the raw text corpus.

The raw sources in this repo include a mix of Project Gutenberg books,
verse-numbered religious texts, and translator/editor footnotes. This module
provides a conservative cleaning pipeline that removes the most common noise
without attempting heavy normalization.
"""

from __future__ import annotations

import re


GUTENBERG_START_RE = re.compile(
	r"(?:\*\*\*\s*)?START OF(?: THE)? PROJECT GUTENBERG EBOOK.*?(?:\*\*\*|$)",
	re.IGNORECASE | re.DOTALL,
)
GUTENBERG_END_RE = re.compile(
	r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
	re.IGNORECASE | re.DOTALL,
)
BRACKETED_NOTE_RE = re.compile(r"\[.*?\]")
VERSE_MARKER_RE = re.compile(r"\b\d+:\d+\b")
CONTENT_START_RE = re.compile(
	r"(?i)^\s*(?:"
	r"(?:chapter|book|section|canto|part|sura|surah)\s+[ivxlcdm\d]+\.?\s*|"
	r"(?:epistle|gospel|psalm|psalms|genesis|exodus|leviticus|numbers|deuteronomy|joshua|judges|ruth|samuel|kings|chronicles|ezra|nehemiah|esther|job|proverbs|ecclesiastes|song of solomon|isaiah|jeremiah|lamentations|ezekiel|daniel|hosea|joel|amos|obadiah|jonah|micah|nahum|habakkuk|zephaniah|haggai|zechariah|malachi|matthew|mark|luke|john|acts|romans|corinthians|galatians|ephesians|philippians|colossians|thessalonians|timothy|titus|philemon|hebrews|james|peter|jude|revelation)\s*$|"
	r"\d+:\d+"
	r")\b.*$"
)
FRONT_MATTER_LINE_RE = re.compile(
	r"(?i)^\s*(?:"
	r"\*\*\*\s*start of(?: the)? project gutenberg ebook.*|"
	r"project gutenberg ebook.*|"
	r"copyright laws are changing.*|"
	r"this header should be the first thing seen.*|"
	r"please read the [\"']legal small print[\"'].*|"
	r"welcome to the world of free plain vanilla electronic texts.*|"
	r"ebooks readable by both humans and by computers.*|"
	r"these ebooks were prepared by thousands of volunteers.*|"
		"sura number.*|chapter number.*|book number.*|table of contents.*|"
	r"title:.*|release date:.*|last updated:.*|language:.*|character set encoding:.*|edition:.*|translator:.*|editor:.*|produced by.*|"
	r"preface\b.*|foreword\b.*|introduction\b.*|contents\b.*|index\b.*|table of contents\b.*|"
	r"by the editor and translator\b.*|"
	r"[0-9]+\s+[0-9]+\s+.*"
	r")\s*$"
)
STANDALONE_CHAPTER_RE = re.compile(
	r"(?im)^\s*(chapter|book|section|canto|part)\s+[ivxlcdm\d]+\b\.?\s*$"
)
STANDALONE_LABEL_RE = re.compile(
	r"(?im)^\s*(footnotes?|notes?|miscellaneous|translator(?:'s)?(?:\s+note)?|by the editor and translator)\s*:??\s*$"
)
LABEL_PREFIX_RE = re.compile(r"(?im)^\s*(translator|editor)\s*:\s*.*$")
MULTISPACE_RE = re.compile(r"[ \t]+")
TRAILING_SPACE_RE = re.compile(r"[ \t]+\n")
LEADING_SPACE_RE = re.compile(r"(?m)^[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")


def strip_gutenberg_boilerplate(text: str) -> str:
	"""Return text from the Gutenberg start marker forward, if present."""

	start_match = GUTENBERG_START_RE.search(text)
	if start_match:
		text = text[start_match.end():]

	end_match = GUTENBERG_END_RE.search(text)
	if end_match:
		text = text[:end_match.start()]

	return text


def strip_leading_front_matter(text: str) -> str:
	"""Drop title pages, tables of contents, and similar leading boilerplate."""

	lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
	content_start_index = None

	for index, line in enumerate(lines):
		stripped_line = line.strip()
		if not stripped_line:
			continue

		if FRONT_MATTER_LINE_RE.match(stripped_line):
			continue

		if CONTENT_START_RE.match(stripped_line):
			content_start_index = index
			break

		# Title pages and tables of contents are usually short, title-cased, and
		# lack sentence punctuation, so keep skipping them until real content.
		continue

	if content_start_index is None:
		return text

	return "\n".join(lines[content_start_index:])


def remove_bracketed_notes(text: str) -> str:
	"""Remove translator notes and footnotes enclosed in square brackets."""

	return BRACKETED_NOTE_RE.sub("", text)


def remove_verse_and_chapter_markers(text: str) -> str:
	"""Remove verse numbers and simple chapter/section headings."""

	text = VERSE_MARKER_RE.sub("", text)
	text = STANDALONE_CHAPTER_RE.sub("", text)
	text = STANDALONE_LABEL_RE.sub("", text)
	text = LABEL_PREFIX_RE.sub("", text)
	return text


def normalize_whitespace(text: str) -> str:
	"""Collapse repeated whitespace while preserving paragraph breaks."""

	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = MULTISPACE_RE.sub(" ", text)
	text = TRAILING_SPACE_RE.sub("\n", text)
	text = LEADING_SPACE_RE.sub("", text)
	text = MULTIBLANK_RE.sub("\n\n", text)
	return text.strip()


def clean_text(text: str) -> str:
	"""Apply the standard cleaning pipeline for the raw corpus."""

	cleaned_text = strip_gutenberg_boilerplate(text)
	cleaned_text = strip_leading_front_matter(cleaned_text)
	cleaned_text = remove_bracketed_notes(cleaned_text)
	cleaned_text = remove_verse_and_chapter_markers(cleaned_text)
	cleaned_text = normalize_whitespace(cleaned_text)
	return cleaned_text


def sliding_window_words(text: str, window_size: int = 100, stride: int = 25):
    """Yield overlapping word windows from cleaned text."""
    words = text.split()
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be greater than 0")

    total_words = len(words)
    if total_words == 0:
        return

    # If the text is shorter than the window size, yield the whole text as one window
    if total_words <= window_size:
        yield words
        return

    for start_index in range(0, total_words, stride):
        window = words[start_index:start_index + window_size]
        if not window:
            break
        yield window
        
        # Ensure we don't skip the final words if the boundary doesn't align perfectly
        if start_index + window_size >= total_words:
            break


def clean_text_file(input_path: str, output_path: str | None = None) -> str:
	"""Clean a text file and optionally write the result to disk.

	Returns the cleaned text so the function can be used in notebooks or tests.
	"""

	with open(input_path, encoding="utf-8") as handle:
		cleaned_text = clean_text(handle.read())

	if output_path:
		with open(output_path, "w", encoding="utf-8") as handle:
			handle.write(cleaned_text)

	return cleaned_text


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Clean raw text sources.")
	parser.add_argument("input_path", help="Path to the raw text file")
	parser.add_argument(
		"output_path",
		nargs="?",
		default=None,
		help="Optional path to write the cleaned text",
	)
	args = parser.parse_args()

	print(clean_text_file(args.input_path, args.output_path))
