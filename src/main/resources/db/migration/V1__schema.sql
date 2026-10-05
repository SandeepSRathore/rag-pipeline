CREATE EXTENSION IF NOT EXISTS vector;

-- One row per ingested file. fingerprint = sha256(chunker settings + content), so either changing
-- triggers re-ingestion.
CREATE TABLE source_document (
    id          uuid PRIMARY KEY,
    source_path text        NOT NULL UNIQUE,
    title       text        NOT NULL,
    fingerprint text        NOT NULL,
    chunk_count integer     NOT NULL,
    origin      text        NOT NULL CHECK (origin IN ('CORPUS', 'UPLOAD')),
    ingested_at timestamptz NOT NULL DEFAULT now()
);

-- Same columns PgVectorStore 2.0.1 creates itself (id, content, metadata json, embedding), plus a generated
-- full-text column that keyword search (M4) will query. PgVectorStore names its insert columns explicitly,
-- so the extra column is invisible to it.
CREATE TABLE vector_store (
    id          uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    content     text,
    metadata    json,
    embedding   vector(1536),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
);

CREATE INDEX vector_store_embedding_hnsw ON vector_store USING hnsw (embedding vector_cosine_ops);
CREATE INDEX vector_store_content_tsv_gin ON vector_store USING gin (content_tsv);
