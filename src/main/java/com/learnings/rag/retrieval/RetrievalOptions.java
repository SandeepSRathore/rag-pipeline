package com.learnings.rag.retrieval;

import java.util.Objects;

import com.learnings.rag.config.RagProperties;

/**
 * Per-call retrieval settings. The defaults come from {@code rag.retrieval.*}; the eval harness overrides them, so
 * one run can compare several configurations and retrieve deeper than the chat endpoint does.
 *
 * @param topK number of chunks to return
 * @param similarityThreshold minimum cosine similarity for vector search; 0 keeps every positive similarity
 * @param mode which retrievers run
 * @param candidates in HYBRID mode, how many chunks each retriever contributes before fusion
 */
public record RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
        if (candidates < 1) {
            throw new IllegalArgumentException("candidates must be at least 1, was " + candidates);
        }
        Objects.requireNonNull(mode, "mode");
    }

    public static RetrievalOptions from(RagProperties properties) {
        RagProperties.Retrieval retrieval = properties.retrieval();
        return new RetrievalOptions(retrieval.topK(), retrieval.similarityThreshold(), retrieval.mode(),
                retrieval.candidates());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates);
    }

    public RetrievalOptions withMode(RetrievalMode mode) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates);
    }
}
