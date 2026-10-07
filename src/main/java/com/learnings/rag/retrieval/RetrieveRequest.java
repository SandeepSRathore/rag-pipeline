package com.learnings.rag.retrieval;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * A retrieval-only request. Every setting but the question is optional and falls back to its {@code rag.retrieval.*}
 * default; the bounds keep a debugging request from asking for unbounded work.
 */
public record RetrieveRequest(@NotBlank @Size(max = 2000) String question,
        RetrievalMode mode,
        @Min(1) @Max(20) Integer topK,
        Boolean rewrite,
        @Min(0) @Max(5) Integer queryVariants,
        Boolean rerank,
        @DecimalMin("0") @DecimalMax("10") Double minScore) {

    RetrievalOptions applyTo(RetrievalOptions defaults) {
        RetrievalOptions options = defaults;
        if (mode != null) {
            options = options.withMode(mode);
        }
        if (topK != null) {
            options = options.withTopK(topK);
        }
        if (rewrite != null) {
            options = options.withRewrite(rewrite);
        }
        if (queryVariants != null) {
            options = options.withQueryVariants(queryVariants);
        }
        if (rerank != null) {
            options = options.withRerank(rerank);
        }
        if (minScore != null) {
            options = options.withMinScore(minScore);
        }
        return options;
    }
}
