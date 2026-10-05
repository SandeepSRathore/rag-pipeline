package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.ingest.CorpusIngestor;

@RagIntegrationTest
class EvalRunnerIT {

    private static final String HNSW_QUESTION = "Which index type is HNSW and how does it build its graph?";

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    EvalRunner runner;

    @Autowired
    RagProperties properties;

    @Autowired
    JdbcClient jdbc;

    @TempDir
    Path corpus;

    @BeforeEach
    void ingestFixtures() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        for (String name : new String[] { "pgvector.adoc", "chat-client.adoc" }) {
            try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
                Files.copy(in, corpus.resolve(name));
            }
        }
        corpusIngestor.ingestDirectory(corpus);
    }

    private static GoldenSet goldenSet(GoldenItem... items) {
        return new GoldenSet(Path.of("eval/golden-set.json"), "0".repeat(64), List.of(items));
    }

    private static GoldenItem item(String id, ExpectedSource expected) {
        return new GoldenItem(id, HNSW_QUESTION, List.of(expected), null, null);
    }

    @Test
    void scoresEveryQuestionAndSummarizesTheConfig() {
        GoldenItem found = item("q01", new ExpectedSource("pgvector.adoc", ""));
        GoldenItem missed = item("q02", new ExpectedSource("chat-client.adoc", "No such section"));

        EvalReport report = runner.run(goldenSet(found, missed), EvalConfig.all(properties));

        assertThat(report.run().goldenItems()).isEqualTo(2);
        assertThat(report.run().index().corpusDocuments()).isEqualTo(2);
        assertThat(report.configs()).singleElement().satisfies(config -> {
            assertThat(config.name()).isEqualTo("vector");
            assertThat(config.items()).extracting(ItemResult::id).containsExactly("q01", "q02");
            assertThat(config.items().get(0).score().firstRelevantRank()).isEqualTo(1);
            assertThat(config.items().get(1).score().firstRelevantRank()).isNull();
            assertThat(config.items()).allSatisfy(result -> assertThat(result.latencyMillis()).isNotNegative());
            assertThat(config.summary().hitAt5()).isEqualTo(0.5);
            assertThat(config.summary().mrrAt10()).isEqualTo(0.5);
        });
    }

    @Test
    void retrievesDeeperThanTheChatEndpointForMrrAt10() {
        EvalReport report = runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties));

        // More than chat's top 5, at most 10. Not min(10, indexed): a fixture chunk sharing no words with the
        // question has cosine 0 under the fake embeddings, and the 0.0 threshold drops it.
        assertThat(report.configs().getFirst().items().getFirst().retrieved())
                .hasSizeGreaterThan(RetrievalMetrics.HIT_K)
                .hasSizeLessThanOrEqualTo(RetrievalMetrics.MRR_K);
    }

    @Test
    void expectedSourcesThatAreNotIndexedFailTheRunInsteadOfScoringZero() {
        GoldenSet typo = goldenSet(item("q01", new ExpectedSource("api/vectordbs/pgvectr.adoc", "")));

        assertThat(runner.unindexedSources(typo)).containsExactly("api/vectordbs/pgvectr.adoc");
        assertThatThrownBy(() -> runner.run(typo, EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("api/vectordbs/pgvectr.adoc");
    }

    @Test
    void anEmptyIndexFailsTheRun() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();

        assertThatThrownBy(() -> runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("POST /api/ingest/corpus");
    }
}
