package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class LexicalOverlapTest {

    @Test
    void findsACopiedPhraseIgnoringCaseAndPunctuation() {
        String chunk = "It builds a multilayer graph. It has better query performance than IVFFlat but slower build times.";

        assertThat(LexicalOverlap.longestSharedRun(
                "Does HNSW give Better Query Performance than IVFFlat?", chunk))
                .isEqualTo("better query performance than ivfflat");
    }

    @Test
    void ownWordsShareOnlyShortRuns() {
        assertThat(LexicalOverlap.wordCount(LexicalOverlap.longestSharedRun(
                "Which setting turns on the multilayer graph index?",
                "The HNSW index type builds a multilayer graph. It uses more memory.")))
                .isEqualTo(2);
    }

    @Test
    void identifiersCountAsOneWord() {
        assertThat(LexicalOverlap.words("Set `spring.ai.vectorstore.pgvector.index-type` on ChatClient.Builder, then build."))
                .containsExactly("set", "spring.ai.vectorstore.pgvector.index-type", "on", "chatclient.builder", "then",
                        "build");
    }

    @Test
    void noSharedWordsGiveAnEmptyRun() {
        assertThat(LexicalOverlap.longestSharedRun("Why?", "Because.")).isEmpty();
    }
}
