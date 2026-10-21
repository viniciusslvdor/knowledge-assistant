"""Generate a grounded answer and validate the model's JSON output."""

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import LLMOutputError
from app.core.models import RetrievedChunk
from app.llm.ollama_client import LLMClient
from app.llm.prompts import (
    INSUFFICIENT_CONTEXT_ANSWER,
    SYSTEM_PROMPT,
    build_retry_message,
    build_user_prompt,
)

logger = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class LLMAnswer(BaseModel):
    """The JSON structure the model must return."""

    answer: str = Field(max_length=4000)
    insufficient_context: bool
    source_ids: list[int] = Field(default_factory=list, max_length=50)


LLM_OUTPUT_SCHEMA = LLMAnswer.model_json_schema()


def parse_llm_output(raw: str, num_sources: int) -> LLMAnswer:
    """Parse and check the model output. Raises LLMOutputError with a short reason."""
    text = _THINK_BLOCK.sub("", raw).strip()
    text = _CODE_FENCE.sub("", text).strip()
    if not text:
        raise LLMOutputError(detail="empty response")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(detail=f"response is not valid JSON ({exc.msg})") from exc
    try:
        parsed = LLMAnswer.model_validate(data, strict=False)
    except ValidationError as exc:
        fields = ", ".join(".".join(map(str, e["loc"])) or "root" for e in exc.errors())
        raise LLMOutputError(detail=f"JSON does not match the schema (fields: {fields})") from exc

    answer = parsed.answer.strip()
    if parsed.insufficient_context:
        # Insufficient answers cite nothing; use a fixed message if the model left it empty.
        return LLMAnswer(
            answer=answer or INSUFFICIENT_CONTEXT_ANSWER, insufficient_context=True, source_ids=[]
        )

    if not answer:
        raise LLMOutputError(detail="answer is empty")
    invalid = [i for i in parsed.source_ids if not 1 <= i <= num_sources]
    if invalid:
        raise LLMOutputError(detail=f"source_ids {invalid} do not exist in the context")
    if not parsed.source_ids:
        raise LLMOutputError(detail="an answer was given but no source_ids were cited")
    unique_ids = list(dict.fromkeys(parsed.source_ids))
    return LLMAnswer(answer=answer, insufficient_context=False, source_ids=unique_ids)


class AnswerGenerator:
    def __init__(self, llm_client: LLMClient, *, max_retries: int = 2):
        self.llm_client = llm_client
        self.max_retries = max_retries

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> LLMAnswer:
        """Ask the model, retrying a bounded number of times on invalid output.

        Raises LLMOutputError if no valid answer is produced; it never makes one up.
        Connection errors are not retried here: they propagate immediately.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, chunks)},
        ]
        last_error: LLMOutputError | None = None
        for attempt in range(1, self.max_retries + 2):
            raw = self.llm_client.chat(messages, LLM_OUTPUT_SCHEMA)
            try:
                return parse_llm_output(raw, num_sources=len(chunks))
            except LLMOutputError as exc:
                last_error = exc
                logger.warning("Invalid LLM output (attempt %d): %s", attempt, exc.detail)
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[:2000]},
                    {"role": "user", "content": build_retry_message(exc.detail or "invalid")},
                ]
        assert last_error is not None
        raise last_error
