package com.learnings.rag.eval;

/**
 * A stored corpus chunk, as the golden-set generator samples it.
 *
 * @param content the contextual header plus body, exactly as embedded
 */
public record CorpusChunk(String sourcePath, String title, String breadcrumb, int chunkIndex, int tokenCount,
        String content) {
}
