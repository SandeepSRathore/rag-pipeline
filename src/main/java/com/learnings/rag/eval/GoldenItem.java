package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section that answers the question (relevant locations, as in IR): hit@5 needs any of
 *        them in the top 5; recall@5 is the share of them found there
 * @param referenceAnswer a short answer written from the source chunk; used by the answer evals in M7
 * @param sourceExcerpt the chunk the question was generated from, as context for reviewers; never scored
 * @param tags optional labels such as {@code identifier}; reports summarize each tag separately
 */
public record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
        String sourceExcerpt, List<String> tags) {

    /** A missing tags field (older files, generated drafts) means untagged. */
    public GoldenItem {
        tags = tags == null ? List.of() : tags;
    }

    public GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
            String sourceExcerpt) {
        this(id, question, expectedSources, referenceAnswer, sourceExcerpt, List.of());
    }
}
