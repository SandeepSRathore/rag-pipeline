package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.document.Document;
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
class KeywordRetrieverIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    KeywordRetriever keywordRetriever;

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

    private static String sourcePath(Document document) {
        return (String) document.getMetadata().get(ChunkMetadata.SOURCE_PATH);
    }

    @Test
    void anExactIdentifierFindsTheChunkThatContainsIt() {
        List<Document> results = keywordRetriever.retrieve("spring.ai.vectorstore.pgvector.index-type", 5);

        assertThat(results).isNotEmpty();
        assertThat(results.getFirst().getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Configuration properties");
        assertThat(results.getFirst().getText()).contains("spring.ai.vectorstore.pgvector.index-type");
    }

    @Test
    void matchesChunksContainingAnyTermNotOnlyAllOfThem() {
        // "zebracorn" occurs in no chunk: with AND semantics this question would match nothing.
        List<Document> results = keywordRetriever.retrieve("How do I stream responses with a zebracorn?", 10);

        assertThat(results).anySatisfy(document -> assertThat(document.getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "chat-client.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Streaming responses"));
    }

    @Test
    void scoresAreTheRankAndResultsComeBestFirst() {
        List<Document> results = keywordRetriever.retrieve("HNSW index graph memory", 10);

        assertThat(results).isNotEmpty();
        assertThat(results).allSatisfy(document -> assertThat(document.getScore()).isPositive());
        assertThat(results).extracting(Document::getScore).isSortedAccordingTo((a, b) -> Double.compare(b, a));
    }

    @Test
    void limitCapsTheResults() {
        assertThat(keywordRetriever.retrieve("spring ai chat client vector store", 2)).hasSizeLessThanOrEqualTo(2);
    }

    @Test
    void stopWordsOnlyReturnNothing() {
        assertThat(keywordRetriever.retrieve("how do I do it?", 10)).isEmpty();
        assertThat(keywordRetriever.retrieve("?!", 10)).isEmpty();
    }

    @Test
    void queryOperatorsInTheQuestionAreTreatedAsText() {
        for (String question : List.of("HNSW & graph | !memory", "it's \"unbalanced", "index:* (type)", "a & | ! b")) {
            assertThat(keywordRetriever.retrieve(question, 5)).as(question).isNotNull();
        }
        assertThat(keywordRetriever.retrieve("HNSW & graph | !memory", 5)).extracting(KeywordRetrieverIT::sourcePath)
                .contains("pgvector.adoc");
    }

    @Test
    void onlySearchesChunksOfTheCurrentEmbeddingModel() {
        RagProperties retiredModel = new RagProperties(properties.corpusDir(), "retired-embedding-model",
                properties.chunking(), properties.retrieval());
        new DocumentIngestionService(vectorStore, documents, new StructureAwareChunker(properties), retiredModel,
                transactionManager)
                .ingest("old.adoc", "old", "= Old\n\nThe zebracorn setting from a retired model.", Origin.CORPUS);

        assertThat(keywordRetriever.retrieve("zebracorn", 10)).isEmpty();
    }
}
