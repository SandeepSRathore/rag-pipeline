package com.learnings.rag.eval;

import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalOptions;

/** A named retrieval configuration compared in an eval run. */
public record EvalConfig(String name, RetrievalOptions options) {

    /** The configurations every run compares. M3 has one; M4–M6 add keyword, hybrid, multi-query and rerank. */
    public static List<EvalConfig> all(RagProperties properties) {
        return List.of(new EvalConfig("vector", RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K)));
    }
}
