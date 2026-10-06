package com.learnings.rag.retrieval;

import com.learnings.rag.config.RagProperties;

/**
 * Per-call retrieval settings. The defaults come from {@code rag.retrieval.*}; the eval harness overrides them, so
 * one run can compare several configurations and retrieve deeper than the chat endpoint does.
 *
 * @param topK number of chunks to return
 * @param similarityThreshold minimum cosine similarity; 0 keeps everything
 */
public record RetrievalOptions(int topK, double similarityThreshold) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
    }

    public static RetrievalOptions from(RagProperties properties) {
        return new RetrievalOptions(properties.retrieval().topK(), properties.retrieval().similarityThreshold());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold);
    }
}
