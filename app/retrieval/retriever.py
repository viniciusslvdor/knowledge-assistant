"""Question -> relevant chunks, with relevance filtering and a context budget."""

import logging

from app.core.models import RetrievedChunk
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Finds chunks relevant to a question.

    The nearest neighbour is not assumed to be relevant: every candidate must
    reach `relevance_threshold` (cosine similarity, higher is closer). The
    surviving chunks are then trimmed to `max_context_chars` so the prompt
    stays small enough for a 4B model's context window.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        *,
        top_k: int,
        relevance_threshold: float,
        max_context_chars: int,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.relevance_threshold = relevance_threshold
        self.max_context_chars = max_context_chars

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if self.vector_store.count() == 0:
            logger.info("Retrieval skipped: knowledge base is empty")
            return []

        query_embedding = self.embedder.embed_query(question)
        candidates = self.vector_store.query(query_embedding, self.top_k)
        relevant = [c for c in candidates if c.similarity >= self.relevance_threshold]

        selected: list[RetrievedChunk] = []
        used_chars = 0
        for chunk in relevant:  # already sorted by similarity, best first
            if used_chars + len(chunk.text) > self.max_context_chars and selected:
                break
            selected.append(chunk)
            used_chars += len(chunk.text)

        logger.info(
            "Retrieved %d candidates, %d above threshold %.2f, %d selected (%d chars); "
            "top similarity=%s",
            len(candidates),
            len(relevant),
            self.relevance_threshold,
            len(selected),
            used_chars,
            f"{candidates[0].similarity:.3f}" if candidates else "n/a",
        )
        return selected
