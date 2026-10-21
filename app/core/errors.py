"""Application errors.

Each error carries a `public_message` that is safe to return to API clients
(no stack traces, paths, or document contents) and the HTTP status to use.
"""


class RAGError(Exception):
    status_code: int = 500
    public_message: str = "An internal error occurred."

    def __init__(self, public_message: str | None = None, *, detail: str | None = None):
        if public_message:
            self.public_message = public_message
        # `detail` is for logs only, never sent to clients.
        self.detail = detail
        super().__init__(detail or self.public_message)


class EmbeddingError(RAGError):
    status_code = 503
    public_message = "The embedding model is unavailable."


class VectorStoreError(RAGError):
    status_code = 503
    public_message = "The knowledge base is unavailable."


class LLMUnavailableError(RAGError):
    status_code = 503
    public_message = "The language model service is unavailable."


class LLMTimeoutError(RAGError):
    status_code = 504
    public_message = "The language model did not respond in time."


class LLMOutputError(RAGError):
    status_code = 502
    public_message = "The language model returned an invalid response. Please try again."
