package com.learnings.rag;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;

import reactor.core.publisher.Flux;

/**
 * Streams pre-scripted chunks, then a usage-only response (like OpenAI with include_usage), and records the
 * prompts it receives.
 */
public class StubChatModel implements ChatModel {

    private final List<String> chunks;
    private final RuntimeException failure;
    private final List<Prompt> prompts = new CopyOnWriteArrayList<>();

    public StubChatModel(String... chunks) {
        this(null, chunks);
    }

    private StubChatModel(RuntimeException failure, String... chunks) {
        this.chunks = List.of(chunks);
        this.failure = failure;
    }

    /** Streams the chunks, then fails with {@code failure} instead of completing. */
    public static StubChatModel failingAfter(RuntimeException failure, String... chunks) {
        return new StubChatModel(failure, chunks);
    }

    public List<Prompt> prompts() {
        return prompts;
    }

    @Override
    public ChatResponse call(Prompt prompt) {
        prompts.add(prompt);
        return new ChatResponse(List.of(new Generation(new AssistantMessage(String.join("", chunks)))));
    }

    @Override
    public Flux<ChatResponse> stream(Prompt prompt) {
        prompts.add(prompt);
        Flux<ChatResponse> tokens = Flux.fromIterable(chunks)
                .map(text -> new ChatResponse(List.of(new Generation(new AssistantMessage(text)))));
        if (failure != null) {
            return tokens.concatWith(Flux.error(failure));
        }
        ChatResponse usage = new ChatResponse(List.of(),
                ChatResponseMetadata.builder().usage(new DefaultUsage(120, 7)).build());
        return tokens.concatWithValues(usage);
    }
}
