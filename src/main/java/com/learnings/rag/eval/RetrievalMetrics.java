package com.learnings.rag.eval;

import java.util.List;

/** Retrieval quality for one golden question, and averaged over a run. Pure functions. */
public final class RetrievalMetrics {

    /** hit@5 and recall@5 look at the first 5 chunks: what the chat endpoint hands the model. */
    public static final int HIT_K = 5;

    /** MRR@10 credits the first relevant chunk anywhere in the first 10. */
    public static final int MRR_K = 10;

    private RetrievalMetrics() {
    }

    /** One retrieved chunk as the metrics see it. @param score cosine similarity, for the report only */
    public record RankedSource(String sourcePath, String breadcrumb, Double score) {
    }

    /**
     * @param firstRelevantRank 1-based rank of the first chunk matching any expected source; null if none was
     *        retrieved
     * @param recallAt5 share of the expected sources matched within the first 5 chunks
     * @param reciprocalRankAt10 1 / firstRelevantRank when it is at most 10, otherwise 0
     */
    public record ItemScore(Integer firstRelevantRank, boolean hitAt5, double recallAt5, double reciprocalRankAt10) {
    }

    public record Summary(int items, double hitAt5, double recallAt5, double mrrAt10, long p50Millis,
            long p95Millis) {
    }

    public static ItemScore score(List<ExpectedSource> expected, List<RankedSource> ranked) {
        Integer firstRelevant = null;
        for (int i = 0; i < ranked.size() && firstRelevant == null; i++) {
            RankedSource chunk = ranked.get(i);
            if (expected.stream().anyMatch(source -> source.matches(chunk.sourcePath(), chunk.breadcrumb()))) {
                firstRelevant = i + 1;
            }
        }
        List<RankedSource> top = ranked.subList(0, Math.min(HIT_K, ranked.size()));
        long covered = expected.stream()
                .filter(source -> top.stream().anyMatch(chunk -> source.matches(chunk.sourcePath(), chunk.breadcrumb())))
                .count();
        boolean hit = firstRelevant != null && firstRelevant <= HIT_K;
        double reciprocalRank = firstRelevant != null && firstRelevant <= MRR_K ? 1.0 / firstRelevant : 0.0;
        return new ItemScore(firstRelevant, hit, (double) covered / expected.size(), reciprocalRank);
    }

    public static Summary summarize(List<ItemScore> scores, List<Long> latenciesMillis) {
        if (scores.isEmpty()) {
            throw new IllegalArgumentException("No scores to summarize");
        }
        return new Summary(scores.size(),
                scores.stream().mapToDouble(score -> score.hitAt5() ? 1 : 0).average().orElseThrow(),
                scores.stream().mapToDouble(ItemScore::recallAt5).average().orElseThrow(),
                scores.stream().mapToDouble(ItemScore::reciprocalRankAt10).average().orElseThrow(),
                percentile(latenciesMillis, 0.50),
                percentile(latenciesMillis, 0.95));
    }

    /** Nearest-rank percentile: the smallest value with at least {@code p} of all values at or below it. */
    public static long percentile(List<Long> values, double p) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("No values");
        }
        List<Long> sorted = values.stream().sorted().toList();
        int index = (int) Math.ceil(p * sorted.size()) - 1;
        return sorted.get(Math.clamp(index, 0, sorted.size() - 1));
    }
}
