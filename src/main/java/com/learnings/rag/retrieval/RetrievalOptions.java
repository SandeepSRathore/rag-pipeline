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
 * @param candidates how many chunks each retriever contributes before fusion (in HYBRID mode and per query variant)
 * @param rewrite rewrite the question with the utility model before searching
 * @param queryVariants extra phrasings searched as well and joined across queries; 0 turns it off
 */
public record RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates,
        boolean rewrite, int queryVariants) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
        if (candidates < 1) {
            throw new IllegalArgumentException("candidates must be at least 1, was " + candidates);
        }
        if (queryVariants < 0) {
            throw new IllegalArgumentException("queryVariants must not be negative, was " + queryVariants);
        }
        Objects.requireNonNull(mode, "mode");
    }

    /** No rewriting and no query variants. */
    public RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates) {
        this(topK, similarityThreshold, mode, candidates, false, 0);
    }

    public static RetrievalOptions from(RagProperties properties) {
        RagProperties.Retrieval retrieval = properties.retrieval();
        return new RetrievalOptions(retrieval.topK(), retrieval.similarityThreshold(), retrieval.mode(),
                retrieval.candidates(), retrieval.rewrite(), retrieval.queryVariants());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withMode(RetrievalMode mode) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withRewrite(boolean rewrite) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withQueryVariants(int queryVariants) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }
}
