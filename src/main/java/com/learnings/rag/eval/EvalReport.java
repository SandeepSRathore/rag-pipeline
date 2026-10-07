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

    /**
     * @param summary hit@5, recall@5, MRR@10 and latency over the answerable questions
     * @param byTag one summary per tag (plus "untagged") over the answerable questions; empty when none is tagged
     * @param refusals questions whose retrieval came back empty, which chat answers without calling the model
     */
    public record ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
            List<TagSummary> byTag, Refusals refusals, List<ItemResult> items) {

        /** Refusals derived from the items. */
        public ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
                List<TagSummary> byTag, List<ItemResult> items) {
            this(name, options, summary, byTag, Refusals.of(items), items);
        }
    }

    /**
     * Questions whose retrieval came back empty: correct for an unanswerable question, a false refusal for an
     * answerable one.
     */
    public record Refusals(int unanswerable, int unanswerableRefused, int answerableRefused) {

        static Refusals of(List<ItemResult> items) {
            int unanswerable = 0;
            int unanswerableRefused = 0;
            int answerableRefused = 0;
            for (ItemResult item : items) {
                boolean refused = item.retrieved().isEmpty();
                if (item.answerable()) {
                    answerableRefused += refused ? 1 : 0;
                }
                else {
                    unanswerable++;
                    unanswerableRefused += refused ? 1 : 0;
                }
            }
            return new Refusals(unanswerable, unanswerableRefused, answerableRefused);
        }
    }

    public record TagSummary(String tag, RetrievalMetrics.Summary summary) {
    }

    /**
     * @param score null for an unanswerable question, which is judged by whether {@code retrieved} is empty
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

        /** Unanswerable items have no expected sources (the golden set file enforces it). */
        public boolean answerable() {
            return !expectedSources.isEmpty();
        }
    }
}
