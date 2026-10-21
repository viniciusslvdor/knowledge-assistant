"""Thin wrapper around the official Ollama client with error mapping."""

import logging
from typing import Protocol

import httpx
import ollama

from app.core.errors import LLMTimeoutError, LLMUnavailableError

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def chat(self, messages: list[dict], json_schema: dict) -> str: ...


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
        context_window: int,
        think: bool | None = False,
        client: ollama.Client | None = None,
    ):
        self.model = model
        self.think = think
        self.options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": context_window,
        }
        self._client = client or ollama.Client(host=base_url, timeout=timeout)

    def chat(self, messages: list[dict], json_schema: dict) -> str:
        """Send a chat request constrained to `json_schema`; return the raw text."""
        kwargs = {}
        if self.think is not None:
            kwargs["think"] = self.think
        try:
            response = self._client.chat(
                model=self.model,
                messages=messages,
                format=json_schema,  # Ollama structured outputs
                options=self.options,
                stream=False,
                **kwargs,
            )
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                raise LLMUnavailableError(
                    "The configured language model is not installed in Ollama.",
                    detail=f"model {self.model!r} not found: {exc.error}",
                ) from exc
            raise LLMUnavailableError(
                detail=f"Ollama error {exc.status_code}: {exc.error}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(detail=f"Ollama request timed out: {exc}") from exc
        except (ConnectionError, httpx.HTTPError) as exc:
            raise LLMUnavailableError(detail=f"Cannot reach Ollama: {exc}") from exc

        if response.done_reason == "length":
            logger.warning("LLM output hit the token limit (num_predict); it may be truncated")
        return response.message.content or ""

    def check_health(self) -> tuple[bool, str]:
        """Return (ok, detail): Ollama reachable and the model installed."""
        try:
            installed = {m.model for m in self._client.list().models}
        except Exception as exc:  # health checks must never raise
            logger.warning("Ollama health check failed: %s", exc)
            return False, "Ollama is not reachable"
        wanted = {self.model, f"{self.model}:latest"}
        if installed & wanted:
            return True, "ok"
        return False, f"model {self.model!r} is not installed (run: ollama pull {self.model})"
