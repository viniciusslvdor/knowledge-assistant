"""Shared fixtures. Nothing here downloads a model or contacts Ollama."""

import hashlib
import json
import math
import re
from collections.abc import Callable
from pathlib import Path

import pytest
from chromadb.api.client import SharedSystemClient
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.llm.generator import AnswerGenerator
from app.main import create_app
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import ChromaVectorStore
from app.services.rag_service import RAGService

DIM = 256
_STOPWORDS = {"the", "and", "for", "are", "what", "how", "does", "with", "must", "many", "per"}


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Texts sharing words get high cosine similarity; unrelated texts score ~0.
    That is enough to exercise threshold filtering without a real model.
    """

    def __init__(self):
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3 or word in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        return self._vector(text)


class FakeLLM:
    """Scripted LLM. Each item in `responses` is a string, an exception to
    raise, or a callable(messages) -> str."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], json_schema: dict) -> str:
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("FakeLLM called more times than scripted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages)
        return item


def llm_json(answer: str, source_ids: list[int], insufficient: bool = False) -> str:
    return json.dumps(
        {"answer": answer, "insufficient_context": insufficient, "source_ids": source_ids}
    )


DOCS = {
    "vacation-policy.md": (
        "# Vacation policy\n\n"
        "Full-time employees receive 25 days of paid vacation per year. "
        "Vacation requests must be submitted at least two weeks in advance "
        "through the HR portal."
    ),
    "security.txt": (
        "Passwords must be rotated every 90 days. "
        "Multi-factor authentication is mandatory for remote access to company systems."
    ),
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection="test-collection",
        documents_dir=tmp_path / "docs",
        chunk_size=300,
        chunk_overlap=50,
        retrieval_top_k=4,
        relevance_threshold=0.2,
        max_context_chars=4000,
        max_question_length=200,
        llm_max_retries=2,
    )


@pytest.fixture
def docs_dir(settings: Settings) -> Path:
    settings.documents_dir.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (settings.documents_dir / name).write_text(text, encoding="utf-8")
    return settings.documents_dir


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vector_store(settings: Settings):
    store = ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    yield store
    SharedSystemClient.clear_system_cache()


def make_pipeline(embedder, store, settings: Settings) -> IngestionPipeline:
    return IngestionPipeline(
        embedder,
        store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_file_bytes=int(settings.max_file_size_mb * 1024 * 1024),
        embedding_model_name="fake",
    )


@pytest.fixture
def pipeline(embedder, vector_store, settings) -> IngestionPipeline:
    return make_pipeline(embedder, vector_store, settings)


@pytest.fixture
def indexed_store(pipeline, vector_store, docs_dir) -> ChromaVectorStore:
    pipeline.ingest_directory(docs_dir)
    return vector_store


def make_retriever(embedder, store, settings: Settings) -> Retriever:
    return Retriever(
        embedder,
        store,
        top_k=settings.retrieval_top_k,
        relevance_threshold=settings.relevance_threshold,
        max_context_chars=settings.max_context_chars,
    )


@pytest.fixture
def make_client(settings, embedder) -> Callable[..., tuple[TestClient, FakeLLM]]:
    """Build a TestClient around a real Chroma store + fake embedder + FakeLLM."""

    def _make(store: ChromaVectorStore, *responses, llm_health=None):
        llm = FakeLLM(*responses)
        service = RAGService(
            make_retriever(embedder, store, settings),
            AnswerGenerator(llm, max_retries=settings.llm_max_retries),
            llm_health=llm_health,
        )
        app = create_app(settings=settings, rag_service=service)
        return TestClient(app, raise_server_exceptions=False), llm

    return _make
