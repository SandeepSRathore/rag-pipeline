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

    /** @param byTag one summary per tag (plus "untagged"); empty when no golden item is tagged */
    public record ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
            List<TagSummary> byTag, List<ItemResult> items) {
    }

    public record TagSummary(String tag, RetrievalMetrics.Summary summary) {
    }

    /**
     * @param retrieved the ranked chunks, best first (up to {@link RetrievalMetrics#MRR_K})
     * @param queries the queries actually searched (after rewriting and expansion); empty if not recorded
     */
    public record ItemResult(String id, String question, List<ExpectedSource> expectedSources,
            RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved,
            List<String> queries) {

        public ItemResult(String id, String question, List<ExpectedSource> expectedSources,
                RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved) {
            this(id, question, expectedSources, score, latencyMillis, retrieved, List.of());
        }
    }
}
