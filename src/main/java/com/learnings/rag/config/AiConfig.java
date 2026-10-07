package com.learnings.rag.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration(proxyBeanMethods = false)
public class AiConfig {

    /**
     * The utility client for query rewriting, expansion and reranking (and later judging): a small, fast,
     * deterministic model, separate from the answer client, with no advisors. The answer client is built in
     * AnswerService from its own ChatClient.Builder, which is prototype-scoped, so these options never reach it.
     */
    @Bean
    ChatClient utilityChatClient(ChatClient.Builder builder, RagProperties properties) {
        return builder
                .defaultOptions(ChatOptions.builder()
                        .model(properties.retrieval().utilityModel())
                        .temperature(0.0))
                .build();
    }
}
