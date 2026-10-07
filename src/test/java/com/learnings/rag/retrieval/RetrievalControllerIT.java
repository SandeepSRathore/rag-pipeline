package com.learnings.rag.retrieval;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.ingest.CorpusIngestor;

@RagIntegrationTest
class RetrievalControllerIT {

    @Autowired
    MockMvc mvc;

    @Autowired
    CorpusIngestor corpusIngestor;

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

    private ResultActions retrieve(String json) throws Exception {
        return mvc.perform(post("/api/retrieve").contentType(APPLICATION_JSON).content(json));
    }

    @Test
    void returnsTheChunksAndTheTraceOfTheRequestedMode() throws Exception {
        retrieve("{\"question\":\"spring.ai.vectorstore.pgvector.index-type\",\"mode\":\"KEYWORD\",\"rerank\":false}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks[0].rank").value(1))
                .andExpect(jsonPath("$.chunks[0].sourcePath").value("pgvector.adoc"))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("keyword")));
        retrieve("{\"question\":\"Which index type is HNSW?\",\"mode\":\"VECTOR\",\"rerank\":false,\"topK\":2}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks.length()").value(2))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("vector")));
    }

    @Test
    void aQuestionAloneUsesTheChatDefaults() throws Exception {
        // Hybrid, top 5, rerank on. The test context's stub chat model can't rate, so reranking falls back.
        retrieve("{\"question\":\"Which index type is HNSW and how does it build its graph?\"}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks.length()").value(lessThanOrEqualTo(5)))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("vector", "keyword", "fusion",
                        PipelineTrace.RERANK_FAILED)));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "{\"question\":\"   \"}",
            "{\"question\":\"q\",\"topK\":0}",
            "{\"question\":\"q\",\"topK\":21}",
            "{\"question\":\"q\",\"queryVariants\":6}",
            "{\"question\":\"q\",\"minScore\":10.5}",
            "{\"question\":\"q\",\"mode\":\"BM25\"}" })
    void outOfRangeOrUnknownSettingsAreRejected(String body) throws Exception {
        retrieve(body).andExpect(status().isBadRequest());
    }
}
