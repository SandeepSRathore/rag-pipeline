package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ExpectedSourceTest {

    private final ExpectedSource indexes = new ExpectedSource("pgvector.adoc", "Indexes");

    @Test
    void matchesTheSectionItselfAndItsSubsections() {
        assertThat(indexes.matches("pgvector.adoc", "Indexes")).isTrue();
        assertThat(indexes.matches("pgvector.adoc", "Indexes › HNSW")).isTrue();
    }

    @Test
    void respectsHeadingBoundaries() {
        assertThat(new ExpectedSource("pgvector.adoc", "Index").matches("pgvector.adoc", "Indexes › HNSW")).isFalse();
        assertThat(indexes.matches("pgvector.adoc", "Auto-Configuration › Indexes")).isFalse();
    }

    @Test
    void otherPagesNeverMatch() {
        assertThat(indexes.matches("qdrant.adoc", "Indexes")).isFalse();
    }

    @Test
    void emptyPrefixAcceptsAnyChunkOfThePage() {
        ExpectedSource page = new ExpectedSource("pgvector.adoc", "");

        assertThat(page.matches("pgvector.adoc", "")).isTrue();
        assertThat(page.matches("pgvector.adoc", "Indexes › HNSW")).isTrue();
    }

    @Test
    void aParentSectionChunkDoesNotCountForADeeperExpectation() {
        assertThat(new ExpectedSource("pgvector.adoc", "Indexes › HNSW").matches("pgvector.adoc", "Indexes")).isFalse();
    }
}
