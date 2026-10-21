"""Environment-based application settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars or a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_parse_none_str="none"
    )

    # Ollama / LLM
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:4b"
    # Qwen "thinking" adds latency without helping short JSON answers.
    # LLM_THINK=none omits the flag for models that reject it.
    llm_think: bool | None = False
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, gt=0)
    llm_context_window: int = Field(default=4096, ge=512)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    request_timeout_seconds: float = Field(default=60.0, gt=0)

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    # When true, models are only loaded from the local Hugging Face cache.
    offline_mode: bool = False

    # Vector store
    chroma_persist_dir: Path = Path("data/chroma")
    chroma_collection: str = "knowledge_base"

    # Ingestion
    documents_dir: Path = Path("data/documents")
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    max_file_size_mb: float = Field(default=20.0, gt=0)

    # Retrieval
    retrieval_top_k: int = Field(default=4, ge=1, le=50)
    # Cosine SIMILARITY in [-1, 1] (similarity = 1 - Chroma cosine distance).
    # Chunks scoring below this value are discarded.
    relevance_threshold: float = Field(default=0.30, ge=-1.0, le=1.0)
    max_context_chars: int = Field(default=6000, ge=500)

    # API
    max_question_length: int = Field(default=1000, ge=10, le=10000)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
