"""Text cleaning and overlapping chunking.

Chunks are measured in characters. all-MiniLM-L6-v2 truncates input after 256
word pieces (roughly 1,000 characters of English), so the default chunk size
of 800 keeps whole chunks inside the model's window.
"""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACE = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def clean_text(text: str) -> str:
    """Normalise whitespace and unicode while keeping paragraphs, lists and
    Markdown headings, which carry meaning."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    lines = [_INLINE_SPACE.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _split_units(text: str, chunk_size: int) -> list[tuple[str, str]]:
    """Break text into (separator, unit) pairs where every unit fits in a chunk.

    Paragraphs are preferred, then sentences, then words, then a hard cut.
    The separator is what joined the unit to the previous one.
    """
    units: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(("\n\n", paragraph))
            continue
        sep = "\n\n"
        for sentence in _SENTENCE_END.split(paragraph):
            if len(sentence) <= chunk_size:
                units.append((sep, sentence))
                sep = " "
                continue
            for word in sentence.split():
                while len(word) > chunk_size:
                    units.append((sep, word[:chunk_size]))
                    sep = ""
                    word = word[chunk_size:]
                if word:
                    units.append((sep, word))
                sep = " "
    return units


def _overlap_tail(text: str, overlap: int) -> str:
    """Last `overlap` characters of text, starting on a word boundary."""
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    match = re.search(r"\s", tail)
    return tail[match.end():].strip() if match else ""


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[str] = []
    current = ""
    for sep, unit in _split_units(text, chunk_size):
        if not current:
            current = unit
            continue
        candidate = current + sep + unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        chunks.append(current)
        tail = _overlap_tail(current, chunk_overlap)
        joiner = sep or " "
        if tail and len(tail) + len(joiner) + len(unit) <= chunk_size:
            current = tail + joiner + unit
        else:
            current = unit
    if current:
        chunks.append(current)
    return chunks
