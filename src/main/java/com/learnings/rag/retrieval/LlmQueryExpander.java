package com.learnings.rag.retrieval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.expansion.QueryExpander;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/**
 * Writes alternative phrasings of a question. Unlike Spring AI's MultiQueryExpander, which splits the reply on
 * newlines and silently returns only the original when the count differs, this asks for structured output and keeps
 * every usable variant: numbering and bullets are stripped; blanks, duplicates and copies of the original are dropped;
 * at most {@code variants} are kept. The original question always comes first, and on any failure it is all that is
 * returned (logged, never thrown).
 */
@Component
public class LlmQueryExpander implements QueryExpander {

    private static final Logger log = LoggerFactory.getLogger(LlmQueryExpander.class);

    /** The structured reply the model is asked for. */
    public record QueryVariants(List<String> queries) {
    }

    private final ChatClient chatClient;
    private final String systemPrompt;
    private final int defaultVariants;

    public LlmQueryExpander(@Qualifier("utilityChatClient") ChatClient utilityChatClient,
            @Value("classpath:prompts/query-expansion.st") Resource systemPrompt, RagProperties properties) {
        this.chatClient = utilityChatClient;
        this.defaultVariants = properties.retrieval().queryVariants();
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public List<Query> expand(Query query) {
        return expand(query, defaultVariants);
    }

    public List<Query> expand(Query query, int variants) {
        if (variants < 1) {
            return List.of(query);
        }
        QueryVariants reply;
        try {
            reply = chatClient
                    .prompt(new Prompt(List.of(new SystemMessage(systemPrompt), new UserMessage(
                            "Write " + variants + " alternative search queries for this question:\n" + query.text()))))
                    .call()
                    .entity(QueryVariants.class);
        }
        catch (RuntimeException e) {
            log.warn("Query expansion failed; searching with the original question only", e);
            return List.of(query);
        }

        List<Query> queries = new ArrayList<>();
        queries.add(query);
        Set<String> seen = new HashSet<>();
        seen.add(normalize(query.text()));
        if (reply != null && reply.queries() != null) {
            for (String candidate : reply.queries()) {
                if (queries.size() > variants) {
                    break;
                }
                String text = candidate == null ? "" : candidate.strip().replaceFirst("^(\\d+[.)]|[-*•])\\s*", "").strip();
                if (!text.isEmpty() && seen.add(normalize(text))) {
                    queries.add(new Query(text));
                }
            }
        }
        if (queries.size() == 1) {
            log.warn("Query expansion produced no usable variant for: {}", query.text());
        }
        return List.copyOf(queries);
    }

    private static String normalize(String text) {
        return text.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }
}
