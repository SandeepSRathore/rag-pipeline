package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.transaction.PlatformTransactionManager;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.CorpusIngestor;
import com.learnings.rag.ingest.DocumentIngestionService;
import com.learnings.rag.ingest.SourceDocument.Origin;
import com.learnings.rag.ingest.SourceDocumentRepository;
import com.learnings.rag.ingest.StructureAwareChunker;

@RagIntegrationTest
class RetrievalPipelineIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    RetrievalPipeline pipeline;

    @Autowired
    JdbcClient jdbc;

    @Autowired
    VectorStore vectorStore;

    @Autowired
    SourceDocumentRepository documents;

    @Autowired
    RagProperties properties;

    @Autowired
    PlatformTransactionManager transactionManager;

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

    @Test
    void ranksThePgvectorChunkFirstForAnIndexQuestion() {
        RetrievalResult result = pipeline.retrieve("Which index type is HNSW and how does it build its graph?");

        assertThat(result.documents()).isNotEmpty().hasSizeLessThanOrEqualTo(5);
        assertThat(result.documents().getFirst().getMetadata()).containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc");
        assertThat(result.documents()).allSatisfy(document -> assertThat(document.getScore()).isNotNull());
        assertThat(result.trace().stages()).singleElement().satisfies(stage -> {
            assertThat(stage.name()).isEqualTo("vector");
            assertThat(stage.hits()).hasSameSizeAs(result.documents());
        });
    }

    @Test
    void topKCanBeOverriddenPerCall() {
        String question = "Which index type is HNSW and how does it build its graph?";
        int indexed = jdbc.sql("SELECT count(*)::int FROM vector_store").query(Integer.class).single();
        assertThat(indexed).as("the fixture corpus must exceed the default top-5").isGreaterThan(5);

        assertThat(pipeline.retrieve(question, RetrievalOptions.from(properties).withTopK(1)).documents()).hasSize(1);
        // Not min(10, indexed): a chunk sharing no words with the question has cosine 0 under the fake embeddings,
        // and the 0.0 threshold (distance < 1) drops it.
        assertThat(pipeline.retrieve(question, RetrievalOptions.from(properties).withTopK(10)).documents())
                .hasSizeGreaterThan(5)
                .hasSizeLessThanOrEqualTo(10);
    }

    @Test
    void keywordModeFindsAnExactIdentifier() {
        RetrievalResult result = pipeline.retrieve("spring.ai.vectorstore.pgvector.index-type",
                RetrievalOptions.from(properties).withMode(RetrievalMode.KEYWORD));

        assertThat(result.documents().getFirst().getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Configuration properties");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("keyword");
    }

    @Test
    void hybridModeFusesBothRetrieversIntoTopK() {
        RetrievalResult result = pipeline.retrieve("Which index type is HNSW and how does it build its graph?",
                RetrievalOptions.from(properties).withMode(RetrievalMode.HYBRID));

        assertThat(result.documents()).isNotEmpty().hasSizeLessThanOrEqualTo(5)
                .extracting(Document::getId).doesNotHaveDuplicates();
        assertThat(result.documents().getFirst().getMetadata()).containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc");
        assertThat(result.documents()).allSatisfy(document ->
                assertThat(document.getScore()).isLessThanOrEqualTo(2.0 / (ReciprocalRankFusion.DEFAULT_K + 1)));
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("vector", "keyword", "fusion");
        assertThat(result.trace().totalMillis()).isNotNegative();
    }

    @Test
    void ignoresChunksEmbeddedByAnotherModel() {
        // e.g. an upload from before an embedding-model switch: its vectors live in a different space.
        RagProperties retiredModel = new RagProperties(properties.corpusDir(), "retired-embedding-model",
                properties.chunking(), properties.retrieval());
        new DocumentIngestionService(vectorStore, documents, new StructureAwareChunker(properties), retiredModel,
                transactionManager)
                .ingest("uploads/hnsw-notes.md", "hnsw-notes",
                        "# HNSW notes\n\nWhich index type is HNSW and how does it build its graph?", Origin.UPLOAD);

        RetrievalResult result = pipeline.retrieve("Which index type is HNSW and how does it build its graph?");

        assertThat(result.documents()).isNotEmpty()
                .extracting(document -> document.getMetadata().get(ChunkMetadata.SOURCE_PATH))
                .doesNotContain("uploads/hnsw-notes.md");
    }

    @Test
    void returnsNothingWhenTheIndexIsEmpty() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();

        assertThat(pipeline.retrieve("anything at all").documents()).isEmpty();
    }
}
