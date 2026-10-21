"""Orchestrates retrieval and generation for one question."""

import logging
import time
from collections.abc import Callable

from app.api.schemas import AskResponse, ComponentHealth, HealthResponse, SourceRef
from app.core.errors import RAGError
from app.core.models import RetrievedChunk
from app.llm.generator import AnswerGenerator
from app.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from app.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


def _to_sources(chunks: list[RetrievedChunk], source_ids: list[int]) -> list[SourceRef]:
    """Map cited source ids (1-based) to unique source references."""
    sources: list[SourceRef] = []
    seen: set[tuple] = set()
    for source_id in source_ids:
        chunk = chunks[source_id - 1]
        key = (chunk.document_id, chunk.chunk_index, chunk.page)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceRef(
                document_id=chunk.document_id,
                filename=chunk.filename,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
            )
        )
    return sources


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        llm_health: Callable[[], tuple[bool, str]] | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self._llm_health = llm_health

    def ask(self, question: str) -> AskResponse:
        started = time.perf_counter()
        chunks = self.retriever.retrieve(question)

        if not chunks:
            # Nothing relevant: answer deterministically without calling the LLM,
            # so the model cannot fill the gap with its own knowledge.
            logger.info("No relevant context; returning insufficient-context answer")
            return AskResponse(
                question=question,
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                sources=[],
                insufficient_context=True,
            )

        result = self.generator.generate(question, chunks)
        sources = [] if result.insufficient_context else _to_sources(chunks, result.source_ids)
        logger.info(
            "Answered in %.2fs: question_chars=%d chunks=%d cited=%d insufficient=%s",
            time.perf_counter() - started,
            len(question),
            len(chunks),
            len(sources),
            result.insufficient_context,
        )
        return AskResponse(
            question=question,
            answer=result.answer,
            sources=sources,
            insufficient_context=result.insufficient_context,
        )

    def health(self) -> HealthResponse:
        try:
            count = self.retriever.vector_store.count()
            store = ComponentHealth(ok=True, detail="ok")
        except RAGError as exc:
            logger.warning("Vector store health check failed: %s", exc.detail)
            count = None
            store = ComponentHealth(ok=False, detail=exc.public_message)

        if self._llm_health is None:
            llm = ComponentHealth(ok=True, detail="not checked")
        else:
            ok, detail = self._llm_health()
            llm = ComponentHealth(ok=ok, detail=detail)

        status = "ok" if store.ok and llm.ok else "degraded"
        return HealthResponse(status=status, vector_store=store, llm=llm, indexed_chunks=count)
