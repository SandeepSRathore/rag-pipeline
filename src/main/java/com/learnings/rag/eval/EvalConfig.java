package com.learnings.rag.eval;

import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalMode;
import com.learnings.rag.retrieval.RetrievalOptions;

/** A named retrieval configuration compared in an eval run. */
public record EvalConfig(String name, RetrievalOptions options) {

    /**
     * The configurations every run compares, all retrieving the top 10 and pinning their rewrite and variant
     * settings. M5 adds rewriting and multi-query (3 variants) on top of hybrid; M6 adds reranking.
     */
    public static List<EvalConfig> all(RagProperties properties) {
        RetrievalOptions base = RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K)
                .withRewrite(false).withQueryVariants(0);
        RetrievalOptions hybrid = base.withMode(RetrievalMode.HYBRID);
        return List.of(
                new EvalConfig("vector", base.withMode(RetrievalMode.VECTOR)),
                new EvalConfig("keyword", base.withMode(RetrievalMode.KEYWORD)),
                new EvalConfig("hybrid", hybrid),
                new EvalConfig("hybrid+rewrite", hybrid.withRewrite(true)),
                new EvalConfig("hybrid+multiquery", hybrid.withQueryVariants(MULTI_QUERY_VARIANTS)),
                new EvalConfig("hybrid+rewrite+multiquery", hybrid.withRewrite(true).withQueryVariants(MULTI_QUERY_VARIANTS)));
    }

    /** The spec's multi-query: the original question plus 3 variants. */
    static final int MULTI_QUERY_VARIANTS = 3;
}
