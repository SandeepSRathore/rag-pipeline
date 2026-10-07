package com.learnings.rag.eval;

import java.util.ArrayList;
import java.util.List;

import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.MinScoreRow;
import com.learnings.rag.eval.EvalReport.Refusals;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.retrieval.LlmReranker;

/**
 * What a reranking configuration would have scored at each minimum score, computed from one run at minimum 0.
 * Reranked chunks come back sorted by rating, so dropping those rated below a minimum removes a suffix of the ranking:
 * the filtered top 10 is exactly what a run at that minimum returns. A question whose rerank failed keeps every chunk,
 * as in the pipeline, where a failed rerank applies no minimum.
 */
final class MinScoreSweep {

    private MinScoreSweep() {
    }

    static List<MinScoreRow> sweep(List<ItemResult> results) {
        List<MinScoreRow> rows = new ArrayList<>();
        for (int minScore = 0; minScore <= LlmReranker.MAX_SCORE; minScore++) {
            List<ItemResult> kept = new ArrayList<>(results.size());
            for (ItemResult item : results) {
                kept.add(keepAtLeast(item, minScore));
            }
            List<ItemResult> answerable = kept.stream().filter(ItemResult::answerable).toList();
            RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(
                    answerable.stream().map(ItemResult::score).toList(),
                    answerable.stream().map(ItemResult::latencyMillis).toList());
            rows.add(new MinScoreRow(minScore, summary.hitAt5(), summary.recallAt5(), summary.mrrAt10(),
                    Refusals.of(kept)));
        }
        return List.copyOf(rows);
    }

    private static ItemResult keepAtLeast(ItemResult item, int minScore) {
        if (item.rerankFellBack()) {
            return item;
        }
        List<RankedSource> kept = item.retrieved().stream()
                .filter(chunk -> chunk.score() == null || chunk.score() >= minScore)
                .toList();
        RetrievalMetrics.ItemScore score = item.answerable() ? RetrievalMetrics.score(item.expectedSources(), kept)
                : null;
        return new ItemResult(item.id(), item.question(), item.expectedSources(), score, item.latencyMillis(), kept,
                item.queries(), false);
    }
}
