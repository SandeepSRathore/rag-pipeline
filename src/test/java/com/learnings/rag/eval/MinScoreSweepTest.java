package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.MinScoreRow;
import com.learnings.rag.eval.EvalReport.Refusals;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;

class MinScoreSweepTest {

    private static final ExpectedSource SOURCE = new ExpectedSource("a.adoc", "");

    private static RankedSource chunk(String page, double rating) {
        return new RankedSource(page, "", rating);
    }

    private static ItemResult answerable(String id, boolean rerankFellBack, RankedSource... ranked) {
        List<RankedSource> list = List.of(ranked);
        return new ItemResult(id, id + "?", List.of(SOURCE), RetrievalMetrics.score(List.of(SOURCE), list), 100, list,
                List.of(id + "?"), rerankFellBack);
    }

    private static ItemResult unanswerable(String id, RankedSource... ranked) {
        return new ItemResult(id, id + "?", List.of(), null, 100, List.of(ranked), List.of(id + "?"), false);
    }

    @Test
    void eachRowDropsTheChunksRatedBelowItsMinimum() {
        List<ItemResult> results = List.of(
                answerable("q1", false, chunk("a.adoc", 9), chunk("b.adoc", 3)), // relevant first, rated 9
                answerable("q2", false, chunk("b.adoc", 6), chunk("a.adoc", 4)), // relevant second, rated 4
                unanswerable("u1", chunk("b.adoc", 2)),
                unanswerable("u2", chunk("b.adoc", 7)));

        List<MinScoreRow> rows = MinScoreSweep.sweep(results);

        assertThat(rows).extracting(MinScoreRow::minScore).containsExactly(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
        assertThat(rows.get(0)).isEqualTo(new MinScoreRow(0, 1.0, 1.0, 0.75, new Refusals(2, 0, 0)));
        assertThat(rows.get(3)).isEqualTo(new MinScoreRow(3, 1.0, 1.0, 0.75, new Refusals(2, 1, 0)));
        assertThat(rows.get(4)).isEqualTo(new MinScoreRow(4, 1.0, 1.0, 0.75, new Refusals(2, 1, 0))); // rated 4 is kept
        assertThat(rows.get(5)).isEqualTo(new MinScoreRow(5, 0.5, 0.5, 0.5, new Refusals(2, 1, 0)));
        assertThat(rows.get(8)).isEqualTo(new MinScoreRow(8, 0.5, 0.5, 0.5, new Refusals(2, 2, 1)));
        assertThat(rows.get(10)).isEqualTo(new MinScoreRow(10, 0.0, 0.0, 0.0, new Refusals(2, 2, 2)));
    }

    @Test
    void aQuestionWhoseRerankFellBackKeepsEveryChunkAtEveryMinimum() {
        // Fused scores (about 0.03) are not ratings; the pipeline applies no minimum after a failed rerank.
        List<MinScoreRow> rows = MinScoreSweep.sweep(List.of(answerable("q1", true, chunk("a.adoc", 0.03))));

        assertThat(rows).allSatisfy(row -> {
            assertThat(row.hitAt5()).isEqualTo(1.0);
            assertThat(row.refusals().answerableRefused()).isZero();
        });
    }
}
