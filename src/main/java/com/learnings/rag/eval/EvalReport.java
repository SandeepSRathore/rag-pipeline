package com.learnings.rag.eval;

import java.time.Instant;
import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalOptions;

/** Everything one eval run measured; written as Markdown and JSON by {@link ReportWriter}. */
public record EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs) {

    /** What was measured against, so two reports can be compared honestly. */
    public record RunInfo(String goldenSet, String goldenSetSha256, int goldenItems, String embeddingModel,
            RagProperties.Chunking chunking, IndexStats index) {
    }

    public record ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
            List<ItemResult> items) {
    }

    /** @param retrieved the ranked chunks, best first (up to {@link RetrievalMetrics#MRR_K}) */
    public record ItemResult(String id, String question, List<ExpectedSource> expectedSources,
            RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved) {
    }
}
