package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section that answers the question (relevant locations, as in IR): hit@5 needs any of
 *        them in the top 5; recall@5 is the share of them found there
 * @param referenceAnswer a short answer written from the source chunk; used by the answer evals in M7
 * @param sourceExcerpt the chunk the question was generated from, as context for reviewers; never scored
 */
public record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
        String sourceExcerpt) {
}
