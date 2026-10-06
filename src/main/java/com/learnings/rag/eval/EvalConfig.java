package com.learnings.rag.eval;

import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalMode;
import com.learnings.rag.retrieval.RetrievalOptions;

/** A named retrieval configuration compared in an eval run. */
public record EvalConfig(String name, RetrievalOptions options) {

    /** The configurations every run compares, all retrieving the top 10. M5–M6 add multi-query and rerank variants. */
    public static List<EvalConfig> all(RagProperties properties) {
        RetrievalOptions base = RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K);
        return List.of(
                new EvalConfig("vector", base.withMode(RetrievalMode.VECTOR)),
                new EvalConfig("keyword", base.withMode(RetrievalMode.KEYWORD)),
                new EvalConfig("hybrid", base.withMode(RetrievalMode.HYBRID)));
    }
}
