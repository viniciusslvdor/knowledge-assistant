"""Embedding models. The same embedder must be used for ingestion and queries."""

import logging
import os
import threading
from typing import Protocol

from app.core.errors import EmbeddingError

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Hugging Face Sentence Transformers embedder, loaded lazily on first use.

    Embeddings are L2-normalised so cosine similarity is well defined.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cpu",
        batch_size: int = 32,
        offline: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.offline = offline
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                if self.offline:
                    # Stop huggingface_hub / transformers from touching the network.
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device, local_files_only=self.offline
                    )
                except Exception as exc:
                    hint = (
                        " Offline mode is on: download the model once with network access."
                        if self.offline
                        else ""
                    )
                    raise EmbeddingError(
                        detail=f"Could not load embedding model {self.model_name}: {exc}.{hint}"
                    ) from exc
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(detail=f"Embedding failed: {exc}") from exc
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]
