"""Vector search, relevance threshold, metadata and persistence."""

from unittest.mock import MagicMock

import pytest
from chromadb.api.client import SharedSystemClient

from app.core.errors import VectorStoreError
from app.core.models import Chunk
from app.retrieval.vector_store import ChromaVectorStore
from tests.conftest import make_pipeline, make_retriever


def test_relevant_chunk_is_retrieved_first(embedder, indexed_store, settings):
    retriever = make_retriever(embedder, indexed_store, settings)
    results = retriever.retrieve("How many vacation days do full-time employees receive?")

    assert results, "expected at least one relevant chunk"
    assert results[0].filename == "vacation-policy.md"
    assert results[0].similarity >= settings.relevance_threshold
    assert all(r.similarity >= settings.relevance_threshold for r in results)
    # Sorted best first
    assert [r.similarity for r in results] == sorted((r.similarity for r in results), reverse=True)


def test_irrelevant_question_returns_nothing(embedder, indexed_store, settings):
    retriever = make_retriever(embedder, indexed_store, settings)
    assert retriever.retrieve("What is the capital of France?") == []


def test_threshold_filters_nearest_neighbour(embedder, indexed_store, settings):
    # A strict threshold drops even the closest match.
    settings_strict = settings.model_copy(update={"relevance_threshold": 0.99})
    retriever = make_retriever(embedder, indexed_store, settings_strict)
    assert retriever.retrieve("vacation days") == []


def test_empty_collection_returns_nothing_without_embedding(embedder, vector_store, settings):
    retriever = make_retriever(embedder, vector_store, settings)
    assert retriever.retrieve("anything") == []
    assert embedder.calls == 0


def test_similarity_is_cosine_similarity(embedder, vector_store):
    vec = embedder.embed_query("multi factor authentication")
    chunk = Chunk("d-0", "d", "a.md", 0, "multi factor authentication", "h", None)
    vector_store.add_chunks([chunk], [vec])
    [hit] = vector_store.query(vec, top_k=5)
    assert hit.similarity == pytest.approx(1.0, abs=1e-4)


def test_metadata_is_preserved(embedder, vector_store):
    chunks = [
        Chunk("doc-1-0", "doc-1", "guide.pdf", 0, "alpha beta gamma", "h", page=3),
        Chunk("doc-1-1", "doc-1", "guide.pdf", 1, "delta epsilon", "h", page=None),
    ]
    vector_store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))

    results = vector_store.query(embedder.embed_query("alpha beta gamma"), top_k=2)
    by_index = {r.chunk_index: r for r in results}
    assert by_index[0].document_id == "doc-1"
    assert by_index[0].filename == "guide.pdf"
    assert by_index[0].page == 3
    assert by_index[0].text == "alpha beta gamma"
    assert by_index[1].page is None


def test_top_k_is_respected(embedder, vector_store, settings):
    chunks = [Chunk(f"d-{i}", "d", "a.md", i, f"vacation note {i}", "h") for i in range(10)]
    vector_store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = make_retriever(
        embedder, vector_store, settings.model_copy(update={"retrieval_top_k": 3})
    )
    assert len(retriever.retrieve("vacation note")) == 3


def test_context_budget_limits_chunks(embedder, vector_store, settings):
    text = "vacation " * 60  # ~540 chars each
    chunks = [Chunk(f"d-{i}", "d", "a.md", i, text + str(i), "h") for i in range(4)]
    vector_store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))
    retriever = make_retriever(
        embedder, vector_store, settings.model_copy(update={"max_context_chars": 1200})
    )
    results = retriever.retrieve("vacation")
    assert len(results) == 2
    assert sum(len(r.text) for r in results) <= 1200


def test_data_persists_across_restarts(embedder, settings, docs_dir):
    store = ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    make_pipeline(embedder, store, settings).ingest_directory(docs_dir)
    count = store.count()
    assert count > 0
    del store
    SharedSystemClient.clear_system_cache()  # drop the in-process client, as a restart would

    reopened = ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection)
    assert reopened.count() == count
    results = make_retriever(embedder, reopened, settings).retrieve("vacation days per year")
    assert results and results[0].filename == "vacation-policy.md"
    SharedSystemClient.clear_system_cache()


def test_chroma_query_failure_is_wrapped(embedder, vector_store):
    vector_store._collection = MagicMock()
    vector_store._collection.count.return_value = 1
    vector_store._collection.query.side_effect = RuntimeError("disk I/O error")
    with pytest.raises(VectorStoreError) as info:
        vector_store.query([0.0] * 256, top_k=3)
    assert "disk I/O error" not in info.value.public_message


def test_unopenable_store_raises_vector_store_error(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    with pytest.raises(VectorStoreError):
        ChromaVectorStore(blocker, "test-collection")
