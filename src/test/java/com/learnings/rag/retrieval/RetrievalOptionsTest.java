package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import com.learnings.rag.config.RagProperties;

class RetrievalOptionsTest {

    private static RagProperties properties(int topK, double threshold, RetrievalMode mode, int candidates) {
        return new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(topK, threshold, mode, candidates, false, 0, "gpt-4.1-mini"));
    }

    @Test
    void defaultsComeFromTheRetrievalProperties() {
        assertThat(RetrievalOptions.from(properties(5, 0.2, RetrievalMode.HYBRID, 20)))
                .isEqualTo(new RetrievalOptions(5, 0.2, RetrievalMode.HYBRID, 20));
    }

    @Test
    void withTopKAndWithModeChangeOnlyThatField() {
        RetrievalOptions options = new RetrievalOptions(5, 0.2, RetrievalMode.VECTOR, 20);

        assertThat(options.withTopK(10)).isEqualTo(new RetrievalOptions(10, 0.2, RetrievalMode.VECTOR, 20));
        assertThat(options.withMode(RetrievalMode.KEYWORD)).isEqualTo(new RetrievalOptions(5, 0.2, RetrievalMode.KEYWORD, 20));
    }

    @Test
    void topKAndCandidatesMustBePositiveAndModeIsRequired() {
        assertThatThrownBy(() -> new RetrievalOptions(0, 0.0, RetrievalMode.VECTOR, 20)).hasMessageContaining("topK");
        assertThatThrownBy(() -> new RetrievalOptions(5, 0.0, RetrievalMode.VECTOR, 0)).hasMessageContaining("candidates");
        assertThatThrownBy(() -> new RetrievalOptions(5, 0.0, null, 20)).hasMessageContaining("mode");
    }

    @Test
    void rewriteAndVariantsDefaultToOffAndCanBeSetPerCall() {
        RetrievalOptions options = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20);

        assertThat(options.rewrite()).isFalse();
        assertThat(options.queryVariants()).isZero();
        assertThat(options.withRewrite(true).withQueryVariants(3))
                .isEqualTo(new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, true, 3));
        assertThatThrownBy(() -> options.withQueryVariants(-1)).hasMessageContaining("queryVariants");
    }
}
