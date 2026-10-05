package com.learnings.rag.ingest;

/** Keys stored on every chunk in {@code vector_store.metadata}. */
public final class ChunkMetadata {

    public static final String SOURCE_ID = "source_id";
    public static final String SOURCE_PATH = "source_path";
    public static final String TITLE = "title";
    public static final String BREADCRUMB = "breadcrumb";
    public static final String CHUNK_INDEX = "chunk_index";
    public static final String TOKEN_COUNT = "token_count";

    private ChunkMetadata() {
    }
}
