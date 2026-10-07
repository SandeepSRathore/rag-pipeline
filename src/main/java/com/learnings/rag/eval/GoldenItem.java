package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section that answers the question (relevant locations, as in IR); empty for an
 *        {@link #UNANSWERABLE} item. hit@5 needs any of
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

    /** The tag of a question the indexed docs don't answer; such an item has no expected sources. */
    public static final String UNANSWERABLE = "unanswerable";

    /** False for an {@link #UNANSWERABLE} item, whose retrieval should come back empty so that chat refuses. */
    public boolean answerable() {
        return !tags.contains(UNANSWERABLE);
    }

    public GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
            String sourceExcerpt) {
        this(id, question, expectedSources, referenceAnswer, sourceExcerpt, List.of());
    }
}
