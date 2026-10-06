package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.TagSummary;
import com.learnings.rag.eval.RetrievalMetrics.ItemScore;

class EvalRunnerTagsTest {

    private static final ExpectedSource SOURCE = new ExpectedSource("a.adoc", "");

    private static GoldenItem item(String id, List<String> tags) {
        return new GoldenItem(id, "Question " + id + "?", List.of(SOURCE), null, null, tags);
    }

    private static ItemResult result(String id, boolean hit) {
        ItemScore score = hit ? new ItemScore(1, true, 1.0, 1.0) : new ItemScore(null, false, 0.0, 0.0);
        return new ItemResult(id, "Question " + id + "?", List.of(SOURCE), score, 100, List.of());
    }

    @Test
    void summarizesEachTagAndTheUntaggedRest() {
        List<GoldenItem> items = List.of(item("q1", List.of()), item("i1", List.of("identifier")),
                item("i2", List.of("identifier")), item("q2", null));
        List<ItemResult> results = List.of(result("q1", true), result("i1", true), result("i2", false),
                result("q2", true));

        List<TagSummary> byTag = EvalRunner.summarizeByTag(items, results);

        assertThat(byTag).extracting(TagSummary::tag).containsExactly("identifier", EvalRunner.UNTAGGED);
        assertThat(byTag.get(0).summary().items()).isEqualTo(2);
        assertThat(byTag.get(0).summary().hitAt5()).isEqualTo(0.5);
        assertThat(byTag.get(1).summary().items()).isEqualTo(2);
        assertThat(byTag.get(1).summary().hitAt5()).isEqualTo(1.0);
    }

    @Test
    void anItemWithSeveralTagsCountsUnderEach() {
        List<TagSummary> byTag = EvalRunner.summarizeByTag(List.of(item("i1", List.of("identifier", "table"))),
                List.of(result("i1", true)));

        assertThat(byTag).extracting(TagSummary::tag).containsExactly("identifier", "table");
    }

    @Test
    void noTaggedItemsMeansNoPerTagSummaries() {
        assertThat(EvalRunner.summarizeByTag(List.of(item("q1", List.of())), List.of(result("q1", true)))).isEmpty();
    }
}
