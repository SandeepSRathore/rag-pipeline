package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import com.learnings.rag.eval.RetrievalMetrics.ItemScore;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.eval.RetrievalMetrics.Summary;

class RetrievalMetricsTest {

    private static final ExpectedSource HNSW = new ExpectedSource("pgvector.adoc", "Indexes › HNSW");
    private static final ExpectedSource STREAMING = new ExpectedSource("chat-client.adoc", "Streaming");

    private static RankedSource chunk(String path, String breadcrumb) {
        return new RankedSource(path, breadcrumb, 0.5);
    }

    private static List<RankedSource> misses(int count) {
        return IntStream.range(0, count).mapToObj(i -> chunk("other.adoc", "Section " + i)).toList();
    }

    private static List<RankedSource> ranked(List<RankedSource> before, RankedSource relevant, int after) {
        List<RankedSource> ranked = new ArrayList<>(before);
        ranked.add(relevant);
        ranked.addAll(misses(after));
        return ranked;
    }

    @Test
    void relevantChunkAtRankThreeIsAHitWithReciprocalRankOneThird() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW),
                ranked(misses(2), chunk("pgvector.adoc", "Indexes › HNSW › Tuning"), 7));

        assertThat(score.firstRelevantRank()).isEqualTo(3);
        assertThat(score.hitAt5()).isTrue();
        assertThat(score.recallAt5()).isEqualTo(1.0);
        assertThat(score.reciprocalRankAt10()).isCloseTo(1.0 / 3, within(1e-9));
    }

    @Test
    void relevantChunkAtRankSevenCountsForMrrButNotForHitOrRecall() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW), ranked(misses(6), chunk("pgvector.adoc", "Indexes › HNSW"), 3));

        assertThat(score.firstRelevantRank()).isEqualTo(7);
        assertThat(score.hitAt5()).isFalse();
        assertThat(score.recallAt5()).isZero();
        assertThat(score.reciprocalRankAt10()).isCloseTo(1.0 / 7, within(1e-9));
    }

    @Test
    void nothingRelevantScoresZero() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW), misses(10));

        assertThat(score).isEqualTo(new ItemScore(null, false, 0.0, 0.0));
    }

    @Test
    void recallCountsEachExpectedSourceCoveredInTheTopFive() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW, STREAMING),
                ranked(List.of(), chunk("pgvector.adoc", "Indexes › HNSW"), 9));

        assertThat(score.firstRelevantRank()).isEqualTo(1);
        assertThat(score.recallAt5()).isEqualTo(0.5);
    }

    @Test
    void anyExpectedSourceCanProvideTheFirstHit() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW, STREAMING),
                ranked(misses(1), chunk("chat-client.adoc", "Streaming"), 8));

        assertThat(score.firstRelevantRank()).isEqualTo(2);
        assertThat(score.hitAt5()).isTrue();
    }

    @Test
    void summaryAveragesItemsAndTakesNearestRankLatencyPercentiles() {
        List<ItemScore> scores = List.of(
                new ItemScore(1, true, 1.0, 1.0),
                new ItemScore(4, true, 0.5, 0.25),
                new ItemScore(null, false, 0.0, 0.0),
                new ItemScore(2, true, 1.0, 0.5));

        Summary summary = RetrievalMetrics.summarize(scores, List.of(120L, 80L, 400L, 100L));

        assertThat(summary.items()).isEqualTo(4);
        assertThat(summary.hitAt5()).isEqualTo(0.75);
        assertThat(summary.recallAt5()).isEqualTo(0.625);
        assertThat(summary.mrrAt10()).isEqualTo(0.4375);
        assertThat(summary.p50Millis()).isEqualTo(100);
        assertThat(summary.p95Millis()).isEqualTo(400);
    }

    @Test
    void percentileOfASingleValueIsThatValue() {
        assertThat(RetrievalMetrics.percentile(List.of(42L), 0.95)).isEqualTo(42);
    }

    @Test
    void summarizingNothingIsAnError() {
        assertThatThrownBy(() -> RetrievalMetrics.summarize(List.of(), List.of()))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
