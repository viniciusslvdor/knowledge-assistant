"""Persistent ChromaDB storage for chunks and their embeddings.

Chroma is used only as a vector store: embeddings are always computed by our
own embedder and passed in explicitly, so Chroma never downloads or runs its
default embedding model. Anonymous telemetry is disabled.

The collection uses cosine distance. Chroma reports `distance = 1 - cosine
similarity`; `query` converts it back to similarity for the caller.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.errors import VectorStoreError
from app.core.models import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    content_hash: str
    chunk_count: int


def _to_metadata(chunk: Chunk) -> dict:
    metadata = {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "content_hash": chunk.content_hash,
    }
    # Chroma metadata cannot hold None, so a missing page is simply omitted.
    if chunk.page is not None:
        metadata["page"] = chunk.page
    return metadata


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, collection_name: str):
        try:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
        except Exception as exc:
            raise VectorStoreError(detail=f"Could not open Chroma at {persist_dir}: {exc}") from exc

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma count failed: {exc}") from exc

    def get_document(self, document_id: str) -> StoredDocument | None:
        try:
            result = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma get failed: {exc}") from exc
        metadatas = result.get("metadatas") or []
        if not metadatas:
            return None
        first = metadatas[0]
        hashes = {m.get("content_hash") for m in metadatas}
        # Mixed hashes mean a previous ingestion was interrupted; report as changed.
        content_hash = first["content_hash"] if len(hashes) == 1 else ""
        return StoredDocument(
            document_id=document_id,
            filename=first["filename"],
            content_hash=content_hash,
            chunk_count=len(metadatas),
        )

    def list_document_ids(self) -> set[str]:
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma get failed: {exc}") from exc
        return {m["document_id"] for m in (result.get("metadatas") or [])}

    def delete_document(self, document_id: str) -> None:
        try:
            self._collection.delete(where={"document_id": document_id})
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma delete failed: {exc}") from exc

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        try:
            # upsert keeps ingestion idempotent if the same chunk id is written twice.
            self._collection.upsert(
                ids=[c.chunk_id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[_to_metadata(c) for c in chunks],
            )
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma upsert failed: {exc}") from exc

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return up to `top_k` nearest chunks, most similar first."""
        try:
            total = self._collection.count()
            if total == 0:
                return []
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(detail=f"Chroma query failed: {exc}") from exc

        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        chunks = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=True):
            page = metadata.get("page")
            chunks.append(
                RetrievedChunk(
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    chunk_index=int(metadata["chunk_index"]),
                    text=text or "",
                    similarity=1.0 - float(distance),
                    page=int(page) if page is not None else None,
                )
            )
        return chunks
