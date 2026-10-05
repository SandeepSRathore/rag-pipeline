# RAG Pipeline

Production-grade retrieval-augmented generation over the Spring AI reference docs: Spring Boot 4.1,
Spring AI 2.0 and Postgres + pgvector. Built milestone by milestone; every retrieval technique is measured
against a golden set (from M3). Design: `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`.

## Run

```bash
export OPENAI_API_KEY=sk-...           # must be valid: ingestion embeds, chat generates
scripts/fetch-corpus.sh                # ~50 Spring AI 2.0.1 pages → corpus/
./mvnw spring-boot:run                 # starts pgvector via compose.yaml
curl -X POST localhost:8081/api/ingest/corpus
open http://localhost:8081
```

`OPENAI_CHAT_MODEL` overrides the chat model (default `gpt-5-mini`). If the UI says an answer could not be
generated, the reason (e.g. an invalid API key) is in the application log.

## API

| Method | Path | |
|---|---|---|
| POST | `/api/ingest/corpus` | Sync `corpus/` into the index (added/updated/skipped/removed) |
| GET | `/api/documents` | Indexed documents |
| POST | `/api/documents` | Upload `.adoc .md .txt .pdf .html .docx` (multipart `file`) |
| DELETE | `/api/documents/{id}` | Remove a document and its chunks |
| POST | `/api/chat` | `{"question": "..."}` → SSE `sources`, `token`…, `done` / `error` |

## Tests

`./mvnw test` runs unit tests (no Docker). `./mvnw verify` also runs the Testcontainers ITs. Tests never call OpenAI.
