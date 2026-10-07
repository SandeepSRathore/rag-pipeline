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
import com.learnings.rag.ingest.DocumentIngestionService;
import com.learnings.rag.ingest.SourceDocument.Origin;
import com.learnings.rag.retrieval.LlmReranker;

@RagIntegrationTest
class EvalRunnerIT {

    private static final String HNSW_QUESTION = "Which index type is HNSW and how does it build its graph?";

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    EvalRunner runner;

    @Autowired
    DocumentIngestionService ingestion;

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
        // A real section, but the question shares no word with any fixture chunk, so nothing is retrieved.
        GoldenItem missed = new GoldenItem("q02", "xylophone", List.of(new ExpectedSource("chat-client.adoc",
                "Streaming responses")), null, null);

        EvalReport report = runner.run(goldenSet(found, missed), EvalConfig.all(properties));

        assertThat(report.run().goldenItems()).isEqualTo(2);
        assertThat(report.run().index().corpusDocuments()).isEqualTo(2);
        assertThat(report.configs()).extracting(EvalReport.ConfigResult::name)
                .containsExactly("vector", "keyword", "hybrid", "hybrid+multiquery", "hybrid+rerank",
                        "hybrid+multiquery+rerank");
        assertThat(report.configs()).allSatisfy(config -> assertThat(config.items()).hasSize(2));
        assertThat(report.configs().getFirst()).satisfies(config -> {
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
    void expectedSourcesThatMatchNoIndexedChunkFailTheRunInsteadOfScoringZero() {
        ingestion.ingest("empty.adoc", "empty", "= Empty\n", Origin.CORPUS); // a document row with 0 chunks
        GoldenSet labels = goldenSet(
                item("q01", new ExpectedSource("api/vectordbs/pgvectr.adoc", "")),
                item("q02", new ExpectedSource("pgvector.adoc", "Configuration properties > HNSW index")),
                item("q03", new ExpectedSource("pgvector.adoc", "Configuration properties ")),
                item("q04", new ExpectedSource("empty.adoc", "")),
                item("q05", new ExpectedSource("pgvector.adoc", "Configuration properties › HNSW index")));

        assertThat(runner.unmatchableSources(labels)).containsExactly(
                "q01: api/vectordbs/pgvectr.adoc",
                "q02: pgvector.adoc › Configuration properties > HNSW index",
                "q03: pgvector.adoc › Configuration properties ",
                "q04: empty.adoc");
    }

    @Test
    void aMistypedSectionFailsTheRunInsteadOfScoringZero() {
        GoldenSet asciiSeparator = goldenSet(item("q01",
                new ExpectedSource("pgvector.adoc", "Configuration properties > HNSW index")));

        assertThatThrownBy(() -> runner.run(asciiSeparator, EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("q01: pgvector.adoc › Configuration properties > HNSW index");
    }

    @Test
    void aPageWithoutChunksFailsTheRunInsteadOfScoringZero() {
        ingestion.ingest("empty.adoc", "empty", "= Empty\n", Origin.CORPUS);

        assertThatThrownBy(() -> runner.run(goldenSet(item("q01", new ExpectedSource("empty.adoc", ""))),
                EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("q01: empty.adoc");
    }

    @Test
    void anEmptyIndexFailsTheRun() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();

        assertThatThrownBy(() -> runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("POST /api/ingest/corpus");
    }

    @Test
    void itemsRecordTheQueriesThatWereSearched() {
        EvalReport report = runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties));

        assertThat(report.configs()).filteredOn(config -> config.name().equals("hybrid"))
                .singleElement().satisfies(config -> assertThat(config.items().getFirst().queries())
                        .containsExactly(HNSW_QUESTION));
        // The stub chat model answers "Stub answer [1].", which is no list of variants: expansion falls back.
        assertThat(report.configs()).filteredOn(config -> config.name().equals("hybrid+multiquery"))
                .singleElement().satisfies(config -> assertThat(config.items().getFirst().queries())
                        .containsExactly(HNSW_QUESTION));
    }

    @Test
    void unanswerableQuestionsAreCountedAsRefusalsNotScored() {
        GoldenItem answerable = item("q01", new ExpectedSource("pgvector.adoc", ""));
        // No word in common with the fixtures: keyword search finds nothing and the fake embeddings score cosine 0.
        GoldenItem unanswerable = new GoldenItem("u01", "xylophone", List.of(), null, null,
                List.of(GoldenItem.UNANSWERABLE));

        EvalReport report = runner.run(goldenSet(answerable, unanswerable), EvalConfig.all(properties));

        // Rewriting searches the stub model's reply ("Stub answer [1].") instead of the question, so it is left out.
        assertThat(report.configs()).filteredOn(config -> !config.options().rewrite()).isNotEmpty()
                .allSatisfy(config -> {
                    assertThat(config.summary().items()).isEqualTo(1);
                    assertThat(config.refusals()).isEqualTo(new EvalReport.Refusals(1, 1, 0));
                    assertThat(config.items().get(1).score()).isNull();
                });
    }

    @Test
    void rerankRowsRecordFallbacksAndSweepTheMinimumScore() {
        EvalReport report = runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties));

        // The stub chat model's reply is no list of ratings, so every rerank falls back to the fused order.
        assertThat(report.configs()).filteredOn(config -> config.options().rerank()).hasSize(2).allSatisfy(config -> {
            assertThat(config.items()).allSatisfy(item -> assertThat(item.rerankFellBack()).isTrue());
            assertThat(config.minScoreSweep()).hasSize(LlmReranker.MAX_SCORE + 1);
        });
        assertThat(report.configs()).filteredOn(config -> !config.options().rerank())
                .allSatisfy(config -> assertThat(config.minScoreSweep()).isEmpty());
        assertThat(retrieved(report, "hybrid+rerank")).isEqualTo(retrieved(report, "hybrid"));
    }

    private static List<RetrievalMetrics.RankedSource> retrieved(EvalReport report, String config) {
        return report.configs().stream().filter(result -> result.name().equals(config)).findFirst().orElseThrow()
                .items().getFirst().retrieved();
    }
}
