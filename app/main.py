"""FastAPI application factory.

Run with: uv run uvicorn app.main:app
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import RAGError
from app.core.logging import configure_logging
from app.llm.generator import AnswerGenerator
from app.llm.ollama_client import OllamaClient
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


def build_rag_service(settings: Settings) -> RAGService:
    """Wire the production dependencies together."""
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        offline=settings.offline_mode,
    )
    vector_store = ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    retriever = Retriever(
        embedder,
        vector_store,
        top_k=settings.retrieval_top_k,
        relevance_threshold=settings.relevance_threshold,
        max_context_chars=settings.max_context_chars,
    )
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.request_timeout_seconds,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        context_window=settings.llm_context_window,
        think=settings.llm_think,
    )
    generator = AnswerGenerator(llm, max_retries=settings.llm_max_retries)
    return RAGService(retriever, generator, llm_health=llm.check_health)


def create_app(settings: Settings | None = None, rag_service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        if getattr(app.state, "rag_service", None) is None:
            app.state.rag_service = build_rag_service(settings)
            health = app.state.rag_service.health()
            if health.status != "ok":
                logger.warning(
                    "Started in degraded mode: vector_store=%s llm=%s",
                    health.vector_store.detail,
                    health.llm.detail,
                )
            elif health.indexed_chunks == 0:
                logger.warning("Knowledge base is empty. Run: uv run python -m app.ingestion")
        yield

    app = FastAPI(
        title="Knowledge Assistant API",
        description="Answers questions from a local knowledge base using Ollama and ChromaDB.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rag_service = rag_service

    @app.exception_handler(RAGError)
    async def handle_rag_error(request: Request, exc: RAGError) -> JSONResponse:
        logger.error("%s on %s: %s", type(exc).__name__, request.url.path, exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.public_message})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    app.include_router(router)
    return app


app = create_app()
