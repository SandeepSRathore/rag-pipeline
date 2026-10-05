package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section needed to answer; hit@5 needs one of them, recall@5 counts how many
 * @param referenceAnswer a short answer written from the source chunk; used by the answer evals in M7
 * @param sourceExcerpt the chunk the question was generated from, as context for reviewers; never scored
 */
public record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
        String sourceExcerpt) {
}
