package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class RetrieveRequestTest {

    private static final RetrievalOptions DEFAULTS = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, false, 0,
            true, 6);

    @Test
    void missingSettingsKeepTheDefaults() {
        assertThat(new RetrieveRequest("q", null, null, null, null, null, null).applyTo(DEFAULTS)).isEqualTo(DEFAULTS);
    }

    @Test
    void givenSettingsOverrideTheDefaults() {
        assertThat(new RetrieveRequest("q", RetrievalMode.VECTOR, 10, true, 3, false, 2.5).applyTo(DEFAULTS))
                .isEqualTo(new RetrievalOptions(10, 0.0, RetrievalMode.VECTOR, 20, true, 3, false, 2.5));
    }
}
