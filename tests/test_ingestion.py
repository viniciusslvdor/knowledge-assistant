"""Loading, cleaning, chunking and idempotent indexing."""

from itertools import pairwise
from pathlib import Path

import pytest

from app.core.errors import EmbeddingError
from app.ingestion.chunking import clean_text, split_text
from app.ingestion.loaders import make_document_id

# --- cleaning & chunking --------------------------------------------------


def test_clean_text_normalises_whitespace_but_keeps_structure():
    raw = "# Title\r\n\r\n\r\n\r\nSome   text\twith\x00 spaces.  \n- item one\n- item two\n"
    assert clean_text(raw) == "# Title\n\nSome text with spaces.\n- item one\n- item two"


def test_split_short_text_is_one_chunk():
    assert split_text("Just one paragraph.", 200, 20) == ["Just one paragraph."]


def test_split_respects_size_and_overlaps():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(40)]
    text = " ".join(sentences)
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    for previous, current in pairwise(chunks):
        # The start of each chunk repeats the end of the previous one.
        assert current[:20] in previous
    # No sentence is lost.
    joined = " ".join(chunks)
    assert all(s in joined for s in sentences)


def test_split_prefers_paragraph_boundaries():
    paragraphs = ["A" * 80, "B" * 80, "C" * 80]
    chunks = split_text("\n\n".join(paragraphs), chunk_size=170, chunk_overlap=0)
    assert chunks == ["A" * 80 + "\n\n" + "B" * 80, "C" * 80]


def test_split_hard_cuts_very_long_words():
    chunks = split_text("x" * 450, chunk_size=200, chunk_overlap=20)
    assert all(len(c) <= 200 for c in chunks)
    assert "".join(chunks).count("x") >= 450


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("text", 100, 100)


def test_document_id_is_stable_and_case_insensitive():
    assert make_document_id("Guides/Setup.md") == make_document_id("guides/setup.md")
    assert make_document_id("a.md") != make_document_id("b.md")


# --- pipeline -------------------------------------------------------------


def test_ingest_directory_indexes_documents_with_metadata(pipeline, vector_store, docs_dir):
    report = pipeline.ingest_directory(docs_dir)

    assert sorted(report.added) == ["security.txt", "vacation-policy.md"]
    assert report.failed == []
    assert vector_store.count() == report.chunks_written > 0
    stored = vector_store.get_document(make_document_id("vacation-policy.md"))
    assert stored is not None and stored.filename == "vacation-policy.md"


def test_reingesting_unchanged_documents_does_not_duplicate(
    pipeline, vector_store, docs_dir, embedder
):
    pipeline.ingest_directory(docs_dir)
    count, embed_calls = vector_store.count(), embedder.calls

    report = pipeline.ingest_directory(docs_dir)

    assert sorted(report.unchanged) == ["security.txt", "vacation-policy.md"]
    assert report.added == report.updated == []
    assert vector_store.count() == count
    assert embedder.calls == embed_calls  # nothing re-embedded


def test_modified_document_replaces_old_chunks(pipeline, vector_store, docs_dir):
    pipeline.ingest_directory(docs_dir)
    (docs_dir / "security.txt").write_text("Badges must be worn at all times.", encoding="utf-8")

    report = pipeline.ingest_directory(docs_dir)

    assert report.updated == ["security.txt"]
    results = vector_store._collection.get(
        where={"document_id": make_document_id("security.txt")}, include=["documents"]
    )
    assert results["documents"] == ["Badges must be worn at all times."]


def test_empty_unsupported_and_unreadable_files_are_reported(pipeline, vector_store, docs_dir):
    (docs_dir / "empty.md").write_text("   \n\n  ", encoding="utf-8")
    (docs_dir / "image.png").write_bytes(b"\x89PNG\r\n")
    (docs_dir / "broken.pdf").write_bytes(b"this is not a pdf")
    (docs_dir / "binary.txt").write_bytes(b"abc\x00\x01\x02")
    (docs_dir / ".hidden.md").write_text("secret notes", encoding="utf-8")

    report = pipeline.ingest_directory(docs_dir)

    skipped = dict(report.skipped)
    failed = dict(report.failed)
    assert skipped["empty.md"] == "no extractable text"
    assert skipped["image.png"] == "unsupported file type"
    assert "broken.pdf" in failed
    assert "binary.txt" in failed
    assert ".hidden.md" not in skipped and ".hidden.md" not in failed
    assert sorted(report.added) == ["security.txt", "vacation-policy.md"]


def test_oversized_file_is_skipped(pipeline, docs_dir):
    pipeline.max_file_bytes = 50
    report = pipeline.ingest_directory(docs_dir)
    assert {name for name, _ in report.skipped} == {"security.txt", "vacation-policy.md"}


def test_prune_removes_deleted_documents(pipeline, vector_store, docs_dir):
    pipeline.ingest_directory(docs_dir)
    (docs_dir / "security.txt").unlink()

    report = pipeline.ingest_directory(docs_dir, prune=True)

    assert report.removed == [make_document_id("security.txt")]
    assert vector_store.get_document(make_document_id("security.txt")) is None
    assert vector_store.get_document(make_document_id("vacation-policy.md")) is not None


def test_nested_folders_keep_relative_filenames(pipeline, vector_store, docs_dir):
    (docs_dir / "guides").mkdir()
    (docs_dir / "guides" / "setup.md").write_text("Install the agent first.", encoding="utf-8")
    report = pipeline.ingest_directory(docs_dir)
    assert "guides/setup.md" in report.added


def test_embedding_failure_keeps_previous_version(pipeline, vector_store, docs_dir, embedder):
    pipeline.ingest_directory(docs_dir)
    count = vector_store.count()
    (docs_dir / "security.txt").write_text("New content about badges.", encoding="utf-8")

    def broken(texts):
        raise EmbeddingError(detail="model crashed")

    embedder.embed_documents = broken
    with pytest.raises(EmbeddingError):
        pipeline.ingest_directory(docs_dir)
    assert vector_store.count() == count


def test_missing_directory_raises(pipeline, tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        pipeline.ingest_directory(tmp_path / "nope")


def test_cli_reports_and_exit_code(tmp_path, monkeypatch, docs_dir):
    """The CLI wires real settings; swap the embedder for the fake one."""
    from chromadb.api.client import SharedSystemClient

    from app.core.config import Settings
    from app.ingestion import __main__ as cli
    from tests.conftest import FakeEmbedder

    cli_settings = Settings(_env_file=None, chroma_persist_dir=tmp_path / "cli-chroma")
    monkeypatch.setattr(cli, "get_settings", lambda: cli_settings)
    monkeypatch.setattr(cli, "SentenceTransformerEmbedder", lambda *a, **k: FakeEmbedder())
    try:
        assert cli.main(["--dir", str(docs_dir)]) == 0
        (docs_dir / "broken.pdf").write_bytes(b"garbage")
        assert cli.main(["--dir", str(docs_dir)]) == 1
        assert cli.main(["--dir", str(tmp_path / "missing")]) == 2
    finally:
        SharedSystemClient.clear_system_cache()
