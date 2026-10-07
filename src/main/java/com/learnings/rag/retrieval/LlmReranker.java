package com.learnings.rag.retrieval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

/**
 * Rates how well each retrieved chunk answers the question, with one structured-output call to the utility model,
 * and reorders the chunks by that rating. Passages are numbered 1..n in the prompt (never by chunk id) and escaped, so
 * a chunk can't close its tag or pose as another passage. Ratings are cleaned rather than trusted: unknown and
 * repeated ids are ignored, scores are clamped to 0–{@value #MAX_SCORE}, and a chunk left unrated scores 0. A failed
 * call, or a reply without one usable rating, comes back empty so the caller can keep the fused order (logged, never
 * thrown).
 */
@Component
public class LlmReranker implements DocumentPostProcessor {

    /** Ratings run from 0 (unrelated) to this (answers the question directly). */
    public static final int MAX_SCORE = 10;

    private static final Logger log = LoggerFactory.getLogger(LlmReranker.class);

    /** Any opening or closing passage(s) tag, in any case and with stray whitespace ("< /PASSAGE >"). */
    private static final Pattern PASSAGE_TAG = Pattern.compile("(?i)<(\\s*/?\\s*passage)");

    /** The structured reply the model is asked for. */
    public record Ratings(List<Rating> ratings) {
    }

    /** @param id the passage number shown in the prompt, from 1 */
    public record Rating(int id, double score) {
    }

    private final ChatClient chatClient;
    private final String systemPrompt;

    public LlmReranker(@Qualifier("utilityChatClient") ChatClient utilityChatClient,
            @Value("classpath:prompts/rerank.st") Resource systemPrompt) {
        this.chatClient = utilityChatClient;
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Spring AI's post-retrieval hook: the documents best first, or unchanged when rating fails. */
    @Override
    public List<Document> process(Query query, List<Document> documents) {
        return rerank(query.text(), documents).orElse(documents);
    }

    /**
     * @return every candidate, best first, each scored with its rating (ties keep the incoming order); empty when the
     *         model failed or returned no usable rating
     */
    public Optional<List<Document>> rerank(String question, List<Document> candidates) {
        if (candidates.isEmpty()) {
            return Optional.of(List.of());
        }
        Ratings reply;
        try {
            reply = chatClient
                    .prompt(new Prompt(List.of(new SystemMessage(systemPrompt),
                            new UserMessage(userMessage(question, candidates)))))
                    .call()
                    .entity(Ratings.class);
        }
        catch (RuntimeException e) {
            log.warn("Reranking failed; keeping the fused order", e);
            return Optional.empty();
        }

        Map<Integer, Double> scores = new HashMap<>();
        if (reply != null && reply.ratings() != null) {
            for (Rating rating : reply.ratings()) {
                if (rating != null && rating.id() >= 1 && rating.id() <= candidates.size()) {
                    scores.putIfAbsent(rating.id(), Math.clamp(rating.score(), 0.0, MAX_SCORE));
                }
            }
        }
        if (scores.isEmpty()) {
            log.warn("Reranking returned no usable rating for {} candidates; keeping the fused order",
                    candidates.size());
            return Optional.empty();
        }
        if (scores.size() < candidates.size()) {
            log.warn("Reranking rated {} of {} candidates; the rest score 0", scores.size(), candidates.size());
        }

        List<Document> rated = new ArrayList<>(candidates.size());
        for (int i = 0; i < candidates.size(); i++) {
            Document candidate = candidates.get(i);
            rated.add(Document.builder()
                    .id(candidate.getId())
                    .text(candidate.getText())
                    .metadata(candidate.getMetadata())
                    .score(scores.getOrDefault(i + 1, 0.0))
                    .build());
        }
        rated.sort(Comparator.comparingDouble(Document::getScore).reversed()); // stable: ties keep the incoming order
        return Optional.of(List.copyOf(rated));
    }

    private static String userMessage(String question, List<Document> candidates) {
        StringBuilder user = new StringBuilder("Question: ").append(question).append("\n\n<passages>\n");
        for (int i = 0; i < candidates.size(); i++) {
            user.append("<passage id=\"").append(i + 1).append("\">\n")
                    .append(PASSAGE_TAG.matcher(candidates.get(i).getText()).replaceAll("&lt;$1"))
                    .append("\n</passage>\n");
        }
        return user.append("</passages>\n\nRate all ").append(candidates.size()).append(" passages.").toString();
    }
}
