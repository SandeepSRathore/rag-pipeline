package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import com.learnings.rag.config.RagProperties;

class RetrievalOptionsTest {

    private static RagProperties properties(int topK, double threshold) {
        return new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(topK, threshold));
    }

    @Test
    void defaultsComeFromTheRetrievalProperties() {
        assertThat(RetrievalOptions.from(properties(5, 0.2))).isEqualTo(new RetrievalOptions(5, 0.2));
    }

    @Test
    void withTopKKeepsTheThreshold() {
        assertThat(new RetrievalOptions(5, 0.2).withTopK(10)).isEqualTo(new RetrievalOptions(10, 0.2));
    }

    @Test
    void topKMustBePositive() {
        assertThatThrownBy(() -> new RetrievalOptions(0, 0.0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("topK");
    }
}
