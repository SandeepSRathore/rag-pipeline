package com.learnings.rag.generation;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class CitationValidatorTest {

    @Test
    void citationsOfExistingSourcesAreValid() {
        CitationValidator.Check check = CitationValidator.check("HNSW is the default [1]. IVFFlat builds faster [2][3].", 3);

        assertThat(check.cited()).containsExactly(1, 2, 3);
        assertThat(check.outOfRange()).isEmpty();
        assertThat(check.valid()).isTrue();
    }

    @Test
    void aCitationWithNoMatchingSourceIsInvalid() {
        CitationValidator.Check check = CitationValidator.check("HNSW [1], IVFFlat [7], none [0].", 5);

        assertThat(check.outOfRange()).containsExactly(7, 0);
        assertThat(check.valid()).isFalse();
    }

    @Test
    void anAnswerThatCitesNothingIsInvalid() {
        assertThat(CitationValidator.check("HNSW is the default index type.", 3).valid()).isFalse();
    }

    @Test
    void numbersInCodeAreNotCitations() {
        String answer = "Read `parts[1]` and see [2].\n```java\nint first = values[0];\n```\nAlso [1].";

        assertThat(CitationValidator.check(answer, 2).cited()).containsExactly(2, 1);
    }

    @Test
    void anUnterminatedCodeFenceHidesTheRestOfTheAnswer() {
        assertThat(CitationValidator.check("Use it [1].\n```yaml\nlist: [9]", 1).cited()).containsExactly(1);
    }

    @Test
    void aRepeatedCitationCountsOnce() {
        assertThat(CitationValidator.check("[2] first, [2] again, then [1].", 2).cited()).containsExactly(2, 1);
    }
}
