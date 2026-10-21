"""Load -> clean -> chunk -> embed -> store, with change detection."""

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from app.core.errors import EmbeddingError, VectorStoreError
from app.core.models import Chunk
from app.ingestion.chunking import clean_text, split_text
from app.ingestion.loaders import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    LoadedDocument,
    load_document,
    make_document_id,
)
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestionReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (filename, reason)
    removed: list[str] = field(default_factory=list)  # document ids pruned
    chunks_written: int = 0

    def summary(self) -> str:
        return (
            f"added={len(self.added)} updated={len(self.updated)} "
            f"unchanged={len(self.unchanged)} skipped={len(self.skipped)} "
            f"failed={len(self.failed)} removed={len(self.removed)} "
            f"chunks_written={self.chunks_written}"
        )


class IngestionPipeline:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: ChromaVectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_file_bytes: int,
        embedding_model_name: str = "",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_file_bytes = max_file_bytes
        self.embedding_model_name = embedding_model_name

    def _fingerprint(self, document: LoadedDocument, cleaned: list[str]) -> str:
        """Hash of the content plus everything that affects the stored vectors.
        Changing chunking or the embedding model re-indexes the document."""
        h = hashlib.sha256()
        h.update(f"{self.chunk_size}|{self.chunk_overlap}|{self.embedding_model_name}".encode())
        for page, text in zip(document.pages, cleaned, strict=True):
            h.update(f"\x1e{page.page}\x1f".encode())
            h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _build_chunks(
        self, document: LoadedDocument, cleaned: list[str], fingerprint: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page, text in zip(document.pages, cleaned, strict=True):
            for piece in split_text(text, self.chunk_size, self.chunk_overlap):
                index = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.document_id}-{index}",
                        document_id=document.document_id,
                        filename=document.filename,
                        chunk_index=index,
                        text=piece,
                        content_hash=fingerprint,
                        page=page.page,
                    )
                )
        return chunks

    def discover_files(self, directory: Path) -> list[Path]:
        base = directory.resolve()
        files = []
        for path in sorted(directory.rglob("*")):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(directory)
            if any(part.startswith(".") for part in relative.parts):
                continue
            # Guard against paths that resolve outside the documents directory.
            if not path.resolve().is_relative_to(base):
                continue
            files.append(path)
        return files

    def ingest_file(self, path: Path, base_dir: Path, report: IngestionReport) -> str:
        """Ingest one file and record the outcome. Returns its document ID."""
        relative = path.resolve().relative_to(base_dir.resolve()).as_posix()
        document_id = make_document_id(relative)

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report.skipped.append((relative, "unsupported file type"))
            return document_id
        if path.stat().st_size > self.max_file_bytes:
            report.skipped.append((relative, f"file larger than {self.max_file_bytes} bytes"))
            return document_id

        try:
            document = load_document(path, base_dir)
        except DocumentLoadError as exc:
            logger.warning("Could not load %s: %s", relative, exc)
            report.failed.append((relative, str(exc)))
            return document_id

        cleaned = [clean_text(page.text) for page in document.pages]
        existing = self.vector_store.get_document(document_id)
        if not any(cleaned):
            if existing:
                self.vector_store.delete_document(document_id)
            report.skipped.append((relative, "no extractable text"))
            return document_id

        fingerprint = self._fingerprint(document, cleaned)
        if existing and existing.content_hash == fingerprint:
            report.unchanged.append(relative)
            return document_id

        chunks = self._build_chunks(document, cleaned, fingerprint)
        # Embed before deleting old chunks so an embedding failure keeps the old version.
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        if existing:
            self.vector_store.delete_document(document_id)
        self.vector_store.add_chunks(chunks, embeddings)
        report.chunks_written += len(chunks)
        (report.updated if existing else report.added).append(relative)
        logger.info("Indexed %s (%d chunks)", relative, len(chunks))
        return document_id

    def ingest_directory(self, directory: Path, *, prune: bool = False) -> IngestionReport:
        """Ingest every supported file under `directory`.

        With `prune=True`, documents in the store that no longer exist under
        `directory` are deleted.
        """
        if not directory.is_dir():
            raise FileNotFoundError(f"Documents directory not found: {directory}")
        report = IngestionReport()
        seen: set[str] = set()
        for path in self.discover_files(directory):
            try:
                seen.add(self.ingest_file(path, directory, report))
            except (EmbeddingError, VectorStoreError):
                # Infrastructure failures affect every file; stop instead of
                # reporting the same error for each one.
                raise
            except Exception as exc:  # one bad file must not stop the run
                relative = path.relative_to(directory).as_posix()
                logger.exception("Unexpected error ingesting %s", relative)
                report.failed.append((relative, f"unexpected error ({type(exc).__name__})"))
        if prune:
            for document_id in self.vector_store.list_document_ids() - seen:
                self.vector_store.delete_document(document_id)
                report.removed.append(document_id)
        return report
