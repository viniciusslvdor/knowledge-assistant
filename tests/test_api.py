"""HTTP contract: validation, status codes, schema, error hygiene."""

from unittest.mock import MagicMock

import pytest

from app.api.schemas import AskResponse
from app.core.errors import (
    LLMOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    VectorStoreError,
)
from app.llm.prompts import INSUFFICIENT_CONTEXT_ANSWER
from tests.conftest import llm_json

RESPONSE_KEYS = {"question", "answer", "sources", "insufficient_context"}
SOURCE_KEYS = {"document_id", "filename", "chunk_index", "page"}


def assert_contract(body: dict) -> AskResponse:
    assert set(body) == RESPONSE_KEYS
    for source in body["sources"]:
        assert set(source) == SOURCE_KEYS
    return AskResponse.model_validate(body)


def test_ask_valid_question(make_client, indexed_store):
    client, llm = make_client(indexed_store, llm_json("Employees get 25 days.", [1]))
    response = client.post("/ask", json={"question": "  How many vacation days do employees get? "})

    assert response.status_code == 200
    body = assert_contract(response.json())
    assert body.question == "How many vacation days do employees get?"
    assert body.answer == "Employees get 25 days."
    assert body.insufficient_context is False
    assert [s.filename for s in body.sources] == ["vacation-policy.md"]
    assert len(llm.calls) == 1


@pytest.mark.parametrize("question", ["", "   ", "\n\t  "])
def test_empty_or_whitespace_question_is_rejected(make_client, indexed_store, question):
    client, llm = make_client(indexed_store)
    response = client.post("/ask", json={"question": question})
    assert response.status_code == 422
    assert llm.calls == []


def test_question_longer_than_configured_limit_is_rejected(make_client, indexed_store):
    client, llm = make_client(indexed_store)
    response = client.post("/ask", json={"question": "vacation " * 50})  # > 200 chars
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "question"]
    assert llm.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"content": b"{not json", "headers": {"Content-Type": "application/json"}},
        {"json": {}},
        {"json": {"question": 123}},
        {"json": {"question": None}},
        {"json": ["question"]},
        {"json": {"question": "ok?", "extra": "field"}},
    ],
)
def test_malformed_request_bodies(make_client, indexed_store, kwargs):
    client, _ = make_client(indexed_store)
    assert client.post("/ask", **kwargs).status_code == 422


def test_empty_knowledge_base_returns_insufficient_context(make_client, vector_store):
    client, llm = make_client(vector_store)
    response = client.post("/ask", json={"question": "How many vacation days?"})

    assert response.status_code == 200
    body = assert_contract(response.json())
    assert body.insufficient_context is True
    assert body.sources == []
    assert body.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert llm.calls == []  # the LLM is never asked without evidence


def test_irrelevant_question_returns_insufficient_context(make_client, indexed_store):
    client, llm = make_client(indexed_store)
    response = client.post("/ask", json={"question": "What is the capital of France?"})

    body = assert_contract(response.json())
    assert response.status_code == 200
    assert body.insufficient_context is True
    assert body.sources == []
    assert llm.calls == []


def test_model_reports_insufficient_context(make_client, indexed_store):
    client, _ = make_client(
        indexed_store, llm_json("Not covered by the documents.", [1, 2], insufficient=True)
    )
    response = client.post("/ask", json={"question": "Can vacation days roll over to next year?"})
    body = assert_contract(response.json())
    assert body.insufficient_context is True
    assert body.sources == []  # cited ids are dropped when evidence is insufficient


def test_malformed_llm_output_returns_502_without_answer(make_client, indexed_store):
    client, llm = make_client(indexed_store, "not json", "{\"answer\": 1}", "```nope```")
    response = client.post("/ask", json={"question": "How many vacation days?"})

    assert response.status_code == 502
    assert response.json() == {"detail": LLMOutputError.public_message}
    assert len(llm.calls) == 3  # 1 attempt + 2 bounded retries


def test_ollama_connection_failure_returns_503(make_client, indexed_store):
    client, _ = make_client(
        indexed_store,
        LLMUnavailableError(detail="Cannot reach Ollama: [WinError 10061] at 127.0.0.1:11434"),
    )
    response = client.post("/ask", json={"question": "How many vacation days?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "The language model service is unavailable."}
    assert "11434" not in response.text


def test_ollama_timeout_returns_504(make_client, indexed_store):
    client, _ = make_client(indexed_store, LLMTimeoutError(detail="timed out"))
    response = client.post("/ask", json={"question": "How many vacation days?"})
    assert response.status_code == 504


def test_vector_store_failure_returns_503_without_internals(make_client, settings, embedder):
    store = MagicMock()
    store.count.side_effect = VectorStoreError(detail="sqlite3.OperationalError: C:/internal/db")
    client, _ = make_client(store)

    response = client.post("/ask", json={"question": "How many vacation days?"})
    assert response.status_code == 503
    assert response.json() == {"detail": "The knowledge base is unavailable."}
    assert "internal/db" not in response.text


def test_unexpected_error_returns_generic_500(make_client, indexed_store):
    client, _ = make_client(indexed_store, RuntimeError("Traceback: INTERNAL-DETAIL-MARKER"))
    response = client.post("/ask", json={"question": "How many vacation days?"})
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "INTERNAL-DETAIL-MARKER" not in response.text


def test_health_ok(make_client, indexed_store):
    client, _ = make_client(indexed_store, llm_health=lambda: (True, "ok"))
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_health_degraded_when_llm_down(make_client, indexed_store):
    client, _ = make_client(indexed_store, llm_health=lambda: (False, "Ollama is not reachable"))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["llm"] == {"ok": False, "detail": "Ollama is not reachable"}


def test_openapi_documents_ask_contract(make_client, indexed_store):
    client, _ = make_client(indexed_store)
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["AskResponse"]["properties"]
    assert set(props) == RESPONSE_KEYS
