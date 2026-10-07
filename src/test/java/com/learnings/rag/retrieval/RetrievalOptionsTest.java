package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import com.learnings.rag.config.RagProperties;

class RetrievalOptionsTest {

    private static RagProperties properties(int topK, double threshold, RetrievalMode mode, int candidates) {
        return new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(topK, threshold, mode, candidates, false, 0, "gpt-4.1-mini",
                        new RagProperties.Rerank(false, 0)));
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
                .isEqualTo(new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, true, 3, false, 0));
        assertThatThrownBy(() -> options.withQueryVariants(-1)).hasMessageContaining("queryVariants");
    }

    @Test
    void rerankIsOffByDefaultAndItsMinimumScoreIsValidated() {
        RetrievalOptions options = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20);

        assertThat(options.rerank()).isFalse();
        assertThat(options.minScore()).isZero();
        assertThat(options.withRerank(true).withMinScore(6))
                .isEqualTo(new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, true, 6));
        assertThatThrownBy(() -> options.withMinScore(-1)).hasMessageContaining("minScore");
        assertThatThrownBy(() -> options.withMinScore(10.5)).hasMessageContaining("minScore");
        assertThatThrownBy(() -> options.withMinScore(Double.NaN)).hasMessageContaining("minScore");
    }

    @Test
    void rerankSettingsComeFromTheProperties() {
        RagProperties properties = new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, "gpt-4.1-mini",
                        new RagProperties.Rerank(true, 6)));

        assertThat(RetrievalOptions.from(properties)).satisfies(options -> {
            assertThat(options.rerank()).isTrue();
            assertThat(options.minScore()).isEqualTo(6.0);
        });
    }
}
