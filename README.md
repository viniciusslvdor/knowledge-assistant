# Knowledge Assistant API

A question-answering API that answers **only from your own documents**. Everything runs locally:

- **FastAPI** for the HTTP API
- **Ollama** running `qwen3.5:4b` to write the answers
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **ChromaDB** for persistent vector storage

No API keys, no cloud services. Questions and documents never leave the machine (the only network access is the one-time download of the embedding model, which you can turn off with `OFFLINE_MODE=true` once it is cached).

---

## Contents

1. [Architecture](#architecture)
2. [Project layout](#project-layout)
3. [Setup](#setup)
4. [Ingest documents](#ingest-documents)
5. [Run the API](#run-the-api)
6. [API reference and examples](#api-reference-and-examples)
7. [Configuration](#configuration)
8. [Tests](#tests)
9. [Troubleshooting](#troubleshooting)
10. [Design decisions, limitations and next steps](#design-decisions-limitations-and-next-steps)

---

## Architecture

| Layer | Package | Responsibility |
|---|---|---|
| API | `app/api` | Request validation, response schema, HTTP status codes |
| Service | `app/services` | Orchestrates retrieval → generation → response |
| Retrieval | `app/retrieval` | Embeddings, ChromaDB access, relevance filtering, context budget |
| LLM | `app/llm` | System prompt, Ollama client, output validation and retries |
| Ingestion | `app/ingestion` | Load TXT/MD/PDF, clean, chunk, index, change detection |
| Core | `app/core` | Settings, errors, logging, shared data classes |

Key design decisions:

- **Two guards against unsupported answers.** Retrieval drops chunks below a cosine-similarity threshold; if nothing survives, the API answers "insufficient" *without calling the LLM*. When context is passed, the system prompt still requires the model to say so if the evidence does not answer the question.
- **Citations by number, validated in code.** Each retrieved chunk is shown to the model as `<source id="N">`. The model returns `source_ids`; the server checks every id exists and maps it to real metadata. The model never writes file names or document ids itself, so it cannot invent a source.
- **Structured output plus validation.** Ollama's `format=<JSON schema>` constrains decoding, and the reply is still parsed and validated with Pydantic. Invalid output triggers at most `LLM_MAX_RETRIES` retries, then a `502`. The server never makes up an answer.
- **Idempotent ingestion.** Document ids come from the file's relative path. Each document stores a fingerprint of its text plus the chunking and embedding settings, so re-running ingestion skips unchanged files and replaces changed ones.
- **Ingestion is a CLI, not an HTTP endpoint.** This removes the file-upload attack surface (paths, sizes, content types) and keeps embedding work out of request handling.
- **Dependency injection through an app factory.** `create_app(settings, rag_service)` lets tests swap in a fake embedder and LLM with no global state.

## Project layout

```
.
├── app/
│   ├── main.py                 # app factory, wiring, error handlers
│   ├── api/
│   │   ├── routes.py           # GET /health, POST /ask
│   │   └── schemas.py          # Pydantic request/response models
│   ├── core/
│   │   ├── config.py           # Settings (env vars / .env)
│   │   ├── errors.py           # error types with safe public messages
│   │   ├── logging.py
│   │   └── models.py           # Chunk, RetrievedChunk
│   ├── ingestion/
│   │   ├── __main__.py         # CLI: python -m app.ingestion
│   │   ├── loaders.py          # TXT / Markdown / PDF
│   │   ├── chunking.py         # cleaning + overlapping chunks
│   │   └── pipeline.py         # load → chunk → embed → store, dedup
│   ├── retrieval/
│   │   ├── embeddings.py       # Sentence Transformers embedder
│   │   ├── vector_store.py     # ChromaDB wrapper (cosine)
│   │   └── retriever.py        # threshold + context budget
│   ├── llm/
│   │   ├── prompts.py          # system prompt, delimiters, escaping
│   │   ├── ollama_client.py    # Ollama calls + error mapping
│   │   └── generator.py        # JSON validation + bounded retries
│   └── services/
│       └── rag_service.py      # orchestration, source mapping, health
├── tests/
│   ├── conftest.py             # fake embedder, fake LLM, temp Chroma
│   ├── test_api.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_ingestion.py
│   ├── test_rag_service.py
│   └── integration/
│       └── test_live_ollama.py # optional, needs real Ollama
├── data/                       # local only, ignored by Git
│   ├── documents/              # knowledge_base.txt (the knowledge base)
│   └── chroma/                 # vector DB (created on first ingest)
├── .env.example
├── pyproject.toml / uv.lock    # dependencies (uv)
└── requirements.txt            # same pins, exported for pip users
```

## Setup

### 1. Python dependencies

Requires Python 3.11+. The project uses [uv](https://docs.astral.sh/uv/):

```powershell
uv sync
```

Without uv:

```powershell
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt pytest
```

> `torch` is installed as a dependency of sentence-transformers. On Linux, PyPI's default torch wheel includes CUDA and is large. For a CPU-only install, run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.

### 2. Ollama and the model

Install Ollama from <https://ollama.com/download>, then:

```powershell
ollama --version          # qwen3.5 needs a recent Ollama, see Troubleshooting
ollama pull qwen3.5:4b
ollama list               # should list qwen3.5:4b
```

Ollama runs as a background service on `http://127.0.0.1:11434`.

### 3. Configuration (optional)

```powershell
copy .env.example .env    # Linux/macOS: cp .env.example .env
```

Every setting has a default, so `.env` is only needed to change something. `.env` is ignored by Git; no secrets are required.

### 4. Knowledge base file

The whole `data/` folder is ignored by Git, so a fresh clone has no documents. Create the folder and place the knowledge base in it:

```powershell
New-Item -ItemType Directory -Force data\documents    # Linux/macOS: mkdir -p data/documents
# then copy knowledge_base.txt into data\documents\
```

## Ingest documents

Put `.md`, `.markdown`, `.txt` or `.pdf` files in `data/documents/` (subfolders are fine; hidden files are ignored), then:

```powershell
uv run python -m app.ingestion                 # index data/documents
uv run python -m app.ingestion --dir D:\docs   # index another folder
uv run python -m app.ingestion --prune         # also remove documents deleted from the folder
```

The first run downloads the embedding model (about 90 MB) into the Hugging Face cache. Example output:

```
added=1 updated=0 unchanged=0 skipped=0 failed=0 removed=0 chunks_written=1
```

Running it again only processes new or changed files:

```
added=0 updated=0 unchanged=1 skipped=0 failed=0 removed=0 chunks_written=0
```

Empty, unsupported, oversized and unreadable files are listed as `skipped` or `failed` and do not stop the run. The exit code is `1` if any file failed.

To rebuild the database from scratch, stop the API, delete `data/chroma/` and run ingestion again.

## Run the API

```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Interactive docs: <http://127.0.0.1:8000/docs>
- The API starts even if Ollama is down; `/health` then reports `degraded` and `/ask` returns `503`.

## API reference and examples

### `POST /ask`

Request:

```json
{ "question": "What services does this company offer?" }
```

The question is trimmed, internal whitespace is collapsed, and empty or whitespace-only questions are rejected. The maximum length is `MAX_QUESTION_LENGTH` (default 1000 characters). Unknown fields are rejected.

The responses below are real outputs from `qwen3.5:4b` on `data/documents/knowledge_base.txt`.

**Answered (200):**

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" `
  -d '{\"question\": \"What services does this company offer?\"}'
```

```json
{
  "question": "What services does this company offer?",
  "answer": "This company offers IT staff augmentation services covering backend, frontend, AI, machine learning, DevOps, and QA.",
  "sources": [
    { "document_id": "doc-0004d7fda2a1698e", "filename": "knowledge_base.txt", "chunk_index": 0, "page": null }
  ],
  "insufficient_context": false
}
```

**Insufficient context (200), nothing relevant found.** The LLM is not called:

```json
{
  "question": "Who won the 1998 FIFA World Cup?",
  "answer": "The available information is insufficient to answer this question.",
  "sources": [],
  "insufficient_context": true
}
```

**Insufficient context (200), related text found but it doesn't answer the question.** The model decides:

```json
{
  "question": "Where is this company headquartered?",
  "answer": "The provided context does not state where this company is headquartered.",
  "sources": [],
  "insufficient_context": true
}
```

Document ids are a hash of each file's relative path. `page` is filled for PDFs and `null` otherwise. The first question after starting Ollama can take up to a minute while the model loads; later answers took about 0.5–0.7 s on the test machine.

**Errors.** Bodies are always `{"detail": "..."}` and never contain stack traces, paths or document text.

| Status | When |
|---|---|
| 422 | Empty/whitespace question, question too long, malformed JSON, missing or wrong-type field, extra fields |
| 502 | The LLM returned invalid output after all retries |
| 503 | Ollama unreachable, model not installed, embedding model or ChromaDB unavailable |
| 504 | Ollama did not answer within `REQUEST_TIMEOUT_SECONDS` |
| 500 | Unexpected error (generic message, details only in server logs) |

### `GET /health`

Returns `200` when everything is ready and `503` when degraded:

```json
{
  "status": "ok",
  "vector_store": { "ok": true, "detail": "ok" },
  "llm": { "ok": true, "detail": "ok" },
  "indexed_chunks": 1
}
```

```json
{
  "status": "degraded",
  "vector_store": { "ok": true, "detail": "ok" },
  "llm": { "ok": false, "detail": "model 'qwen3.5:4b' is not installed (run: ollama pull qwen3.5:4b)" },
  "indexed_chunks": 1
}
```

## Configuration

All settings are environment variables (or `.env` entries). See [.env.example](.env.example) for the full list with defaults.

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen3.5:4b` | Generation model |
| `LLM_THINK` | `false` | Qwen thinking mode; `none` = don't send the flag |
| `LLM_TEMPERATURE` | `0.1` | Low for factual answers |
| `LLM_MAX_TOKENS` | `512` | Max generated tokens (`num_predict`) |
| `LLM_CONTEXT_WINDOW` | `4096` | `num_ctx` sent to Ollama |
| `LLM_MAX_RETRIES` | `2` | Extra attempts on invalid JSON |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Timeout for each Ollama call |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Must be the same for ingestion and queries |
| `EMBEDDING_DEVICE` | `cpu` | `cuda` if available |
| `OFFLINE_MODE` | `false` | `true` = only use locally cached models, never download |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Database folder |
| `CHROMA_COLLECTION` | `knowledge_base` | Collection name |
| `DOCUMENTS_DIR` | `data/documents` | Default ingestion folder |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | In characters |
| `MAX_FILE_SIZE_MB` | `20` | Larger files are skipped |
| `RETRIEVAL_TOP_K` | `4` | Candidates fetched from ChromaDB |
| `RELEVANCE_THRESHOLD` | `0.30` | Minimum **cosine similarity** (not distance) |
| `MAX_CONTEXT_CHARS` | `6000` | Cap on retrieved text sent to the LLM |
| `MAX_QUESTION_LENGTH` | `1000` | Characters |
| `LOG_LEVEL` | `INFO` | |

## Tests

```powershell
uv run pytest                 # unit tests: no model, no Ollama, no internet
uv run pytest -v tests/test_api.py
```

The unit tests use a deterministic bag-of-words embedder and a scripted fake LLM, with a real ChromaDB in a temporary folder. They cover validation, empty/irrelevant questions, malformed LLM output and retries, Ollama and ChromaDB failures, citation mapping, persistence across restarts, prompt injection, duplicate ingestion and the response schema.

Optional integration tests run the real embedding model and a real Ollama:

```powershell
$env:RUN_INTEGRATION="1"; uv run pytest tests/integration -v; Remove-Item Env:RUN_INTEGRATION
# bash: RUN_INTEGRATION=1 uv run pytest tests/integration -v
```

They skip themselves if Ollama or the model is not available.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ollama pull qwen3.5:4b` → `412: requires a newer version of Ollama` | Update Ollama from <https://ollama.com/download>, then pull again. Ollama 0.24.0 is too old for qwen3.5. |
| `/ask` → 503 "language model service is unavailable" | Start Ollama (`ollama serve`, or the desktop app) and check `OLLAMA_BASE_URL`. |
| `/ask` → 503 "configured language model is not installed" | `ollama pull <OLLAMA_MODEL>`; check the name with `ollama list`. |
| Ollama error about `think` | The model doesn't support the thinking flag: set `LLM_THINK=none`. |
| `/ask` → 504 | First calls load the model into memory and can be slow on CPU. Raise `REQUEST_TIMEOUT_SECONDS`. |
| `/ask` → 502 often | The model is struggling with the JSON format. Check logs for the reason; try a larger model or raise `LLM_MAX_TOKENS` if answers are being cut off. |
| `/ask` → 503 "embedding model is unavailable" | First run needs internet to download the model. With `OFFLINE_MODE=true`, the model must already be in the Hugging Face cache. |
| Every question returns `insufficient_context: true` | Run ingestion; check `/health` → `indexed_chunks`; lower `RELEVANCE_THRESHOLD`. Questions in languages other than English score low with MiniLM (see limitations). |
| Answers cite unrelated text | Raise `RELEVANCE_THRESHOLD` or lower `RETRIEVAL_TOP_K`. |
| `database is locked` / file errors on Windows | Stop the API before deleting `data/chroma` or running two writers at once. |

## Design decisions, limitations and next steps

### Trade-offs

- **Character-based chunking** (paragraph → sentence → word boundaries) instead of token-based. It is simple and has no extra dependencies. 800 characters stays below MiniLM's 256-token input limit for typical English text.
- **Fixed similarity threshold** instead of a re-ranker. It is cheap and explainable but blunt; see the table above.
- **No retries on connection errors.** If Ollama is down, retrying only delays the 503. Retries are spent only on invalid output.
- **One embedding model per collection.** Mixing models in one collection would make similarities meaningless; changing the model re-indexes everything.
- **Sync route handlers.** FastAPI runs them in a thread pool, so blocking model calls don't stall the event loop. Simple, but throughput is bounded by the local LLM anyway.