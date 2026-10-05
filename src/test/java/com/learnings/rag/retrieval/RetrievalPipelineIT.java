package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.ingest.CorpusIngestor;

@RagIntegrationTest
class RetrievalPipelineIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    RetrievalPipeline pipeline;

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
    void returnsNothingWhenTheIndexIsEmpty() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();

        assertThat(pipeline.retrieve("anything at all").documents()).isEmpty();
    }
}
