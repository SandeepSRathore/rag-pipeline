package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.transaction.PlatformTransactionManager;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.ingest.CorpusIngestor;
import com.learnings.rag.ingest.DocumentIngestionService;
import com.learnings.rag.ingest.SourceDocument.Origin;
import com.learnings.rag.ingest.SourceDocumentRepository;
import com.learnings.rag.ingest.StructureAwareChunker;

@RagIntegrationTest
class ChunkCatalogIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    DocumentIngestionService ingestion;

    @Autowired
    ChunkCatalog catalog;

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
    void ingest() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        for (String name : new String[] { "pgvector.adoc", "chat-client.adoc" }) {
            try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
                Files.copy(in, corpus.resolve(name));
            }
        }
        corpusIngestor.ingestDirectory(corpus);
        ingestion.ingest("uploads/notes.md", "notes", "# Notes\n\nUploaded notes about HNSW tuning.", Origin.UPLOAD);
        RagProperties retiredModel = new RagProperties(properties.corpusDir(), "retired-embedding-model",
                properties.chunking(), properties.retrieval());
        new DocumentIngestionService(vectorStore, documents, new StructureAwareChunker(properties), retiredModel,
                transactionManager)
                .ingest("old.adoc", "old", "= Old\n\nEmbedded with a retired model.", Origin.CORPUS);
    }

    @Test
    void corpusChunksAreTheCurrentModelsCorpusChunksInPageOrder() {
        List<CorpusChunk> chunks = catalog.corpusChunks();

        assertThat(chunks).extracting(CorpusChunk::sourcePath)
                .containsOnly("chat-client.adoc", "pgvector.adoc")
                .isSorted();
        assertThat(chunks).filteredOn(chunk -> chunk.sourcePath().equals("pgvector.adoc"))
                .extracting(CorpusChunk::chunkIndex)
                .isSorted();
        assertThat(chunks).anySatisfy(chunk -> {
            assertThat(chunk.title()).isEqualTo("PGvector");
            assertThat(chunk.breadcrumb()).isEqualTo("Configuration properties › HNSW index");
            assertThat(chunk.content()).startsWith("PGvector › Configuration properties › HNSW index\n\n");
            assertThat(chunk.tokenCount()).isPositive();
        });
    }

    @Test
    void statsCountCorpusPagesUploadsAndChunksForTheCurrentModel() {
        int currentModelChunks = jdbc.sql("SELECT count(*)::int FROM vector_store WHERE metadata->>'"
                + ChunkMetadata.EMBEDDING_MODEL + "' = :model")
                .param("model", properties.embeddingModel())
                .query(Integer.class).single();

        assertThat(catalog.stats()).isEqualTo(new IndexStats(2, 1, currentModelChunks));
    }
}
