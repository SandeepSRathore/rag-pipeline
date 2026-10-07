package com.learnings.rag.eval;

import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalMode;
import com.learnings.rag.retrieval.RetrievalOptions;

/** A named retrieval configuration compared in an eval run. */
public record EvalConfig(String name, RetrievalOptions options) {

    /**
     * The configurations every run compares, all retrieving the top 10 with rewriting off and reranking pinned. M5's
     * rewrite rows are gone: rewriting lost clearly, and eval/README.md keeps their numbers. Multi-query stays as the
     * baseline for multi-query plus rerank. Rerank rows run at minimum score 0; the report sweeps the minimum from
     * their recorded ratings.
     */
    public static List<EvalConfig> all(RagProperties properties) {
        RetrievalOptions base = RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K)
                .withRewrite(false).withQueryVariants(0).withRerank(false).withMinScore(0);
        RetrievalOptions hybrid = base.withMode(RetrievalMode.HYBRID);
        RetrievalOptions multiQuery = hybrid.withQueryVariants(MULTI_QUERY_VARIANTS);
        return List.of(
                new EvalConfig("vector", base.withMode(RetrievalMode.VECTOR)),
                new EvalConfig("keyword", base.withMode(RetrievalMode.KEYWORD)),
                new EvalConfig("hybrid", hybrid),
                new EvalConfig("hybrid+multiquery", multiQuery),
                new EvalConfig("hybrid+rerank", hybrid.withRerank(true)),
                new EvalConfig("hybrid+multiquery+rerank", multiQuery.withRerank(true)));
    }

    /** The spec's multi-query: the original question plus 3 variants. */
    static final int MULTI_QUERY_VARIANTS = 3;
}
