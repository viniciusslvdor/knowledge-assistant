"""Internal data structures shared between layers."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of a document ready to be embedded and stored."""

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    content_hash: str
    page: int | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search, with its cosine similarity."""

    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity: float
    page: int | None = None
