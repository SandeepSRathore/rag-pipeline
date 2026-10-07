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
 * @param candidates how many chunks each retriever contributes before fusion (in HYBRID mode and per query variant),
 *        and how many the reranker rates
 * @param rewrite rewrite the question with the utility model before searching
 * @param queryVariants extra phrasings searched as well and joined across queries; 0 turns it off
 * @param rerank rate the candidates with the utility model and keep the best {@code topK}
 * @param minScore with {@code rerank}: drop candidates rated below this (0–10); ignored without it
 */
public record RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates,
        boolean rewrite, int queryVariants, boolean rerank, double minScore) {

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
        if (!(minScore >= 0 && minScore <= LlmReranker.MAX_SCORE)) { // also rejects NaN
            throw new IllegalArgumentException(
                    "minScore must be between 0 and " + LlmReranker.MAX_SCORE + ", was " + minScore);
        }
        Objects.requireNonNull(mode, "mode");
    }

    /** No rewriting, no query variants and no reranking. */
    public RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates) {
        this(topK, similarityThreshold, mode, candidates, false, 0, false, 0);
    }

    public static RetrievalOptions from(RagProperties properties) {
        RagProperties.Retrieval retrieval = properties.retrieval();
        return new RetrievalOptions(retrieval.topK(), retrieval.similarityThreshold(), retrieval.mode(),
                retrieval.candidates(), retrieval.rewrite(), retrieval.queryVariants(), retrieval.rerank().enabled(),
                retrieval.rerank().minScore());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }

    public RetrievalOptions withMode(RetrievalMode mode) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }

    public RetrievalOptions withRewrite(boolean rewrite) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }

    public RetrievalOptions withQueryVariants(int queryVariants) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }

    public RetrievalOptions withRerank(boolean rerank) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }

    public RetrievalOptions withMinScore(double minScore) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants, rerank, minScore);
    }
}
