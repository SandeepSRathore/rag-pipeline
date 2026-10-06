package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;

class QueryRewriterTest {

    private static QueryRewriter rewriter(ChatModel model) {
        return new QueryRewriter(ChatClient.builder(model).build());
    }

    @Test
    void searchesWithTheModelsRewrite() {
        assertThat(rewriter(new StubChatModel("  pgvector index type options \n"))
                .rewrite("hey so which index thing do I pick for pgvector??"))
                .isEqualTo("pgvector index type options");
    }

    @Test
    void anEmptyReplyKeepsTheOriginalQuestion() {
        assertThat(rewriter(new StubChatModel("")).rewrite("Which index types does PGvector support?"))
                .isEqualTo("Which index types does PGvector support?");
    }

    @Test
    void aFailingModelKeepsTheOriginalQuestion() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        assertThat(rewriter(unavailable).rewrite("Which index types does PGvector support?"))
                .isEqualTo("Which index types does PGvector support?");
    }

    @Test
    void bracesInTheQuestionReachTheModelIntact() {
        StubChatModel model = new StubChatModel("template placeholders");

        rewriter(model).rewrite("How do I fill {name} in a PromptTemplate?");

        assertThat(model.prompts()).singleElement().satisfies(prompt ->
                assertThat(prompt.getUserMessage().getText()).contains("How do I fill {name} in a PromptTemplate?"));
    }
}
