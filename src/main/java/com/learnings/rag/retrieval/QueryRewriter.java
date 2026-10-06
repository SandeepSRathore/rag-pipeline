package com.learnings.rag.retrieval;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.transformation.RewriteQueryTransformer;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/** Rewrites a chatty or vague question into a concise search query with Spring AI's RewriteQueryTransformer. */
@Component
public class QueryRewriter {

    private static final Logger log = LoggerFactory.getLogger(QueryRewriter.class);

    private final RewriteQueryTransformer transformer;

    public QueryRewriter(@Qualifier("utilityChatClient") ChatClient utilityChatClient) {
        this.transformer = RewriteQueryTransformer.builder()
                .chatClientBuilder(utilityChatClient.mutate())
                .targetSearchSystem("search engine over the Spring AI reference documentation")
                .build();
    }

    /** The rewritten question; the original when the model fails or answers with nothing (logged, never thrown). */
    public String rewrite(String question) {
        try {
            String rewritten = transformer.transform(new Query(question)).text();
            return rewritten == null || rewritten.isBlank() ? question : rewritten.strip();
        }
        catch (RuntimeException e) {
            log.warn("Query rewrite failed; searching with the original question", e);
            return question;
        }
    }
}
