package com.learnings.rag.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.pgvector.PgVectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.NestedExceptionUtils;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.transaction.PlatformTransactionManager;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.CorpusIngestor.CorpusReport;
import com.learnings.rag.ingest.DocumentIngestionService.IngestOutcome.Status;
import com.learnings.rag.ingest.SourceDocument.Origin;

@RagIntegrationTest
class IngestionIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    VectorStore vectorStore;

    @Autowired
    SourceDocumentRepository documents;

    @Autowired
    JdbcClient jdbc;

    @Autowired
    JdbcTemplate jdbcTemplate;

    @Autowired
    PlatformTransactionManager transactionManager;

    @Autowired
    RagProperties properties;

    @TempDir
    Path corpus;

    @BeforeEach
    void setUp() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        copyFixture("pgvector.adoc");
        copyFixture("chat-client.adoc");
    }

    @Test
    void firstRunAddsEverythingAndSecondRunSkipsEverything() throws IOException {
        CorpusReport first = corpusIngestor.ingestDirectory(corpus);

        assertThat(first.added()).isEqualTo(2);
        assertThat(first.chunksWritten()).isPositive().isEqualTo(vectorRows());

        CorpusReport second = corpusIngestor.ingestDirectory(corpus);

        assertThat(second).isEqualTo(new CorpusReport(0, 0, 2, 0, 0));
        assertThat(vectorRows()).isEqualTo(first.chunksWritten());
    }

    @Test
    void changedFileReplacesOnlyItsOwnChunks() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        Files.writeString(corpus.resolve("pgvector.adoc"),
                "\n\n== Zebracorn tuning\n\nThe zebracorn setting is fictional and only exists in this test.\n",
                StandardOpenOption.APPEND);

        CorpusReport report = corpusIngestor.ingestDirectory(corpus);

        assertThat(report.updated()).isEqualTo(1);
        assertThat(report.skipped()).isEqualTo(1);
        assertThat(vectorRows()).isEqualTo(sumOfChunkCounts());
        assertThat(rowsContaining("zebracorn")).isPositive();
    }

    @Test
    void fileRemovedFromCorpusIsRemovedFromIndex() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        Files.delete(corpus.resolve("chat-client.adoc"));

        CorpusReport report = corpusIngestor.ingestDirectory(corpus);

        assertThat(report.removed()).isEqualTo(1);
        assertThat(jdbc.sql("SELECT count(*)::int FROM vector_store WHERE metadata->>'source_path' = 'chat-client.adoc'")
                .query(Integer.class).single()).isZero();
        assertThat(documents.findBySourcePath("chat-client.adoc")).isEmpty();
    }

    @Test
    void emptyCorpusDirectoryIsRejectedInsteadOfWipingTheIndex() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        Path empty = Files.createDirectory(corpus.resolve("nothing-here"));

        assertThatThrownBy(() -> corpusIngestor.ingestDirectory(empty)).isInstanceOf(CorpusUnavailableException.class);
        assertThatThrownBy(() -> corpusIngestor.ingestDirectory(corpus.resolve("missing")))
                .isInstanceOf(CorpusUnavailableException.class);
        assertThat(vectorRows()).isPositive();
    }

    @Test
    void changingChunkSettingsReingestsUnchangedFiles() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        DocumentIngestionService retuned = ingestionService(vectorStore,
                new StructureAwareChunker(new RagProperties.Chunking(80, 5, 20)), properties.embeddingModel());

        assertThat(reingestPgvector(retuned)).isEqualTo(Status.UPDATED);
        assertThat(vectorRows()).isEqualTo(sumOfChunkCounts());
    }

    @Test
    void changingTheEmbeddingModelReingestsUnchangedFiles() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        DocumentIngestionService otherModel = ingestionService(vectorStore, new StructureAwareChunker(properties),
                "another-embedding-model");

        assertThat(reingestPgvector(otherModel)).isEqualTo(Status.UPDATED);
        assertThat(vectorRows()).isEqualTo(sumOfChunkCounts());
    }

    @Test
    void failedEmbeddingLeavesThePreviousVersionIntact() throws IOException {
        corpusIngestor.ingestDirectory(corpus);
        int rowsBefore = vectorRows();
        String fingerprintBefore = documents.findBySourcePath("pgvector.adoc").orElseThrow().fingerprint();
        VectorStore failingStore = PgVectorStore.builder(jdbcTemplate, new FailingEmbeddingModel())
                .dimensions(1536)
                .initializeSchema(false)
                .build();
        DocumentIngestionService failing = ingestionService(failingStore, new StructureAwareChunker(properties),
                properties.embeddingModel());

        assertThatThrownBy(() -> failing.ingest("pgvector.adoc", "pgvector", "= PGvector\n\nA zebracorn rewrite.",
                Origin.CORPUS))
                .satisfies(e -> assertThat(NestedExceptionUtils.getMostSpecificCause(e))
                        .hasMessage("embedding service unavailable"));

        assertThat(vectorRows()).isEqualTo(rowsBefore);
        assertThat(rowsContaining("zebracorn")).isZero();
        assertThat(documents.findBySourcePath("pgvector.adoc").orElseThrow().fingerprint()).isEqualTo(fingerprintBefore);
    }

    @Test
    void chunksCarryAContextualHeaderAndMetadata() throws IOException {
        corpusIngestor.ingestDirectory(corpus);

        Map<String, Object> row = jdbc.sql("""
                SELECT content,
                       metadata->>'source_path' AS source_path,
                       metadata->>'breadcrumb'  AS breadcrumb,
                       metadata->>'source_id'   AS source_id
                FROM vector_store WHERE content LIKE '%multilayer graph%'""")
                .query().singleRow();

        assertThat((String) row.get("content")).startsWith("PGvector › Configuration properties › HNSW index\n\n");
        assertThat(row.get("source_path")).isEqualTo("pgvector.adoc");
        assertThat(row.get("breadcrumb")).isEqualTo("Configuration properties › HNSW index");
        assertThat(row.get("source_id")).isNotNull();
    }

    private void copyFixture(String name) throws IOException {
        try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
            Files.copy(in, corpus.resolve(name));
        }
    }

    private int vectorRows() {
        return jdbc.sql("SELECT count(*)::int FROM vector_store").query(Integer.class).single();
    }

    private int sumOfChunkCounts() {
        return jdbc.sql("SELECT coalesce(sum(chunk_count), 0)::int FROM source_document").query(Integer.class).single();
    }

    private int rowsContaining(String word) {
        return jdbc.sql("SELECT count(*)::int FROM vector_store WHERE content ILIKE '%' || :word || '%'")
                .param("word", word).query(Integer.class).single();
    }

    private Status reingestPgvector(DocumentIngestionService service) throws IOException {
        return service.ingest("pgvector.adoc", "pgvector", Files.readString(corpus.resolve("pgvector.adoc")),
                Origin.CORPUS).status();
    }

    private DocumentIngestionService ingestionService(VectorStore store, StructureAwareChunker chunker,
            String embeddingModel) {
        RagProperties withModel = new RagProperties(properties.corpusDir(), embeddingModel, properties.chunking(),
                properties.retrieval());
        return new DocumentIngestionService(store, documents, chunker, withModel, transactionManager);
    }

    /** Simulates OpenAI being unreachable or rejecting the API key. */
    static class FailingEmbeddingModel implements EmbeddingModel {

        @Override
        public EmbeddingResponse call(EmbeddingRequest request) {
            throw new IllegalStateException("embedding service unavailable");
        }

        @Override
        public float[] embed(Document document) {
            throw new IllegalStateException("embedding service unavailable");
        }

        @Override
        public int dimensions() {
            return 1536;
        }
    }
}
