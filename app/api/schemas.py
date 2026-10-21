"""Public request/response contract."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Absolute upper bound; the configurable MAX_QUESTION_LENGTH is checked in the route.
HARD_MAX_QUESTION_LENGTH = 10_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        min_length=1,
        max_length=HARD_MAX_QUESTION_LENGTH,
        examples=["What does the knowledge base say about the topic?"],
    )

    @field_validator("question")
    @classmethod
    def normalise_question(cls, value: str) -> str:
        value = _CONTROL_CHARS.sub("", value)
        value = _WHITESPACE.sub(" ", value).strip()
        if not value:
            raise ValueError("question must not be empty or whitespace only")
        return value


class SourceRef(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    page: int | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceRef]
    insufficient_context: bool


class ComponentHealth(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_store: ComponentHealth
    llm: ComponentHealth
    indexed_chunks: int | None


class ErrorResponse(BaseModel):
    detail: str
