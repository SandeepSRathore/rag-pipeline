package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.rag.Query;
import org.springframework.core.io.ClassPathResource;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.config.RagProperties;

class LlmQueryExpanderTest {

    private static final Query QUESTION = new Query("Which index types does PGvector support?");

    private static LlmQueryExpander expander(ChatModel model) {
        RagProperties properties = new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 3, "gpt-4.1-mini",
                        new RagProperties.Rerank(false, 0)));
        return new LlmQueryExpander(ChatClient.builder(model).build(),
                new ClassPathResource("prompts/query-expansion.st"), properties);
    }

    private static List<String> texts(List<Query> queries) {
        return queries.stream().map(Query::text).toList();
    }

    @Test
    void theOriginalComesFirstThenTheVariants() {
        StubChatModel model = new StubChatModel(
                "{\"queries\": [\"pgvector HNSW vs IVFFlat\", \"spring.ai.vectorstore.pgvector.index-type\", \"postgres vector index options\"]}");

        assertThat(texts(expander(model).expand(QUESTION))).containsExactly(QUESTION.text(),
                "pgvector HNSW vs IVFFlat", "spring.ai.vectorstore.pgvector.index-type", "postgres vector index options");
    }

    @Test
    void cleansMessyVariants() {
        StubChatModel model = new StubChatModel("{\"queries\": [\"1. pgvector HNSW index\", \"\", \"PGvector  hnsw index\", "
                + "\"Which index types does PGvector support?\", \"- index type property\", \"  another phrasing \", \"one too many\"]}");

        assertThat(texts(expander(model).expand(QUESTION, 3))).containsExactly(QUESTION.text(),
                "pgvector HNSW index", "index type property", "another phrasing");
    }

    @Test
    void unparseableReplyMeansTheOriginalOnly() {
        assertThat(texts(expander(new StubChatModel("here are some ideas: a, b, c")).expand(QUESTION)))
                .containsExactly(QUESTION.text());
    }

    @Test
    void aFailingModelMeansTheOriginalOnly() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("model not found");
            }
        };

        assertThat(texts(expander(unavailable).expand(QUESTION))).containsExactly(QUESTION.text());
    }

    @Test
    void zeroVariantsSkipsTheModel() {
        StubChatModel model = new StubChatModel("{\"queries\": [\"x\"]}");

        assertThat(texts(expander(model).expand(QUESTION, 0))).containsExactly(QUESTION.text());
        assertThat(model.prompts()).isEmpty();
    }

    @Test
    void thePromptAsksForTheRequestedNumberAndCarriesTheQuestion() {
        StubChatModel model = new StubChatModel("{\"queries\": []}");

        expander(model).expand(QUESTION, 3);

        assertThat(model.prompts()).singleElement().satisfies(prompt -> {
            assertThat(prompt.getSystemMessage().getText()).contains("Spring AI reference documentation");
            assertThat(prompt.getUserMessage().getText()).contains("Write 3 alternative search queries", QUESTION.text());
        });
    }
}
