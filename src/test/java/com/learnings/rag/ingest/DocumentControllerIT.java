package com.learnings.rag.ingest;

import static org.hamcrest.Matchers.containsString;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import com.jayway.jsonpath.JsonPath;
import com.learnings.rag.RagIntegrationTest;

@RagIntegrationTest
class DocumentControllerIT {

    @Autowired
    MockMvc mvc;

    @Autowired
    JdbcClient jdbc;

    @BeforeEach
    void clean() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
    }

    private static MockMultipartFile file(String name, String content) {
        return new MockMultipartFile("file", name, "application/octet-stream", content.getBytes(StandardCharsets.UTF_8));
    }

    @Test
    void uploadListAndDeleteADocument() throws Exception {
        String body = mvc.perform(multipart("/api/documents")
                        .file(file("notes.md", "# Notes\n\n## Tuning\n\nUse HNSW for low latency search.")))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.status").value("ADDED"))
                .andExpect(jsonPath("$.document.sourcePath").value("uploads/notes.md"))
                .andExpect(jsonPath("$.document.title").value("Notes"))
                .andExpect(jsonPath("$.document.origin").value("UPLOAD"))
                .andReturn().getResponse().getContentAsString();
        String id = JsonPath.read(body, "$.document.id");

        mvc.perform(get("/api/documents"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$[0].sourcePath").value("uploads/notes.md"));
        mvc.perform(delete("/api/documents/{id}", id)).andExpect(status().isNoContent());
        mvc.perform(delete("/api/documents/{id}", id)).andExpect(status().isNotFound());
        mvc.perform(get("/api/documents")).andExpect(jsonPath("$").isEmpty());
    }

    @Test
    void reuploadingIdenticalContentIsSkipped() throws Exception {
        mvc.perform(multipart("/api/documents").file(file("same.md", "Same content."))).andExpect(status().isCreated());

        mvc.perform(multipart("/api/documents").file(file("same.md", "Same content.")))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("SKIPPED"));
    }

    @Test
    void unsupportedFileTypeIs415() throws Exception {
        mvc.perform(multipart("/api/documents").file(file("diagram.png", "not really a png")))
                .andExpect(status().isUnsupportedMediaType());
    }

    @Test
    void fileWithoutTextIs422() throws Exception {
        mvc.perform(multipart("/api/documents").file(file("blank.md", "   \n")))
                .andExpect(status().is(422));
    }

    @Test
    void directoryPartsOfUploadedFilenamesAreDropped() throws Exception {
        mvc.perform(multipart("/api/documents").file(file("../../etc/passwd.md", "Harmless text.")))
                .andExpect(status().isCreated())
                .andExpect(jsonPath("$.document.sourcePath").value("uploads/passwd.md"));
    }

    @Test
    void ingestingAMissingCorpusDirectoryIs409WithAHint() throws Exception {
        mvc.perform(post("/api/ingest/corpus"))
                .andExpect(status().isConflict())
                .andExpect(jsonPath("$.detail", containsString("fetch-corpus.sh")));
    }
}
