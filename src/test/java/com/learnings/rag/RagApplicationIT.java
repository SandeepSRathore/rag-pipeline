package com.learnings.rag;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.test.web.servlet.MockMvc;

@RagIntegrationTest
class RagApplicationIT {

    @Autowired
    JdbcClient jdbc;

    @Autowired
    MockMvc mvc;

    @Test
    void flywayCreatesTheVectorStoreWithAFullTextColumn() {
        List<String> columns = jdbc.sql("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'vector_store' ORDER BY ordinal_position""")
                .query(String.class)
                .list();

        assertThat(columns).containsExactly("id", "content", "metadata", "embedding", "content_tsv");
    }

    @Test
    void healthIsUp() throws Exception {
        mvc.perform(get("/actuator/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"));
    }
}
