package com.learnings.rag;

import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;

/** Replaces the OpenAI models (disabled by the test profile) with offline fakes. */
@TestConfiguration(proxyBeanMethods = false)
public class TestAiConfiguration {

    @Bean
    FakeEmbeddingModel embeddingModel() {
        return new FakeEmbeddingModel();
    }

    @Bean
    StubChatModel chatModel() {
        return new StubChatModel("Stub answer [1].");
    }
}
