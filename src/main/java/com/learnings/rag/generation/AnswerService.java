package com.learnings.rag.generation;

import java.util.List;
import java.util.Locale;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Service;

import com.learnings.rag.retrieval.RetrievalPipeline;
import com.learnings.rag.retrieval.RetrievalResult;

import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

/** Retrieve, then stream a grounded answer. Sources go out first, so the UI can render citations as they arrive. */
@Service
public class AnswerService {

    public static final String NO_SOURCES_ANSWER =
            "I couldn't find anything about that in the indexed documentation. Try rephrasing, or ingest the relevant documents first.";

    public static final String ANSWER_FAILED =
            "Sorry, the answer could not be generated. The server log has the details.";

    /** The refusal sentence the system prompt asks for (prompts/answer-system.st). */
    public static final String PROMPT_REFUSAL = "I couldn't find this in the indexed documentation.";

    private static final List<String> REFUSALS = List.of(
            "i couldn't find this in the indexed documentation",
            "i couldn't find anything about that in the indexed documentation");

    /**
     * Whether an answer is a refusal: the no-sources answer, or the prompt's refusal sentence anywhere in it (models
     * sometimes prefix it, e.g. "Direct answer: I couldn't find this…"). Case and curly apostrophes don't matter.
     */
    public static boolean isRefusal(String answer) {
        String normalized = answer.replace('\u2019', '\'').toLowerCase(Locale.ROOT);
        return REFUSALS.stream().anyMatch(normalized::contains);
    }

    private static final Logger log = LoggerFactory.getLogger(AnswerService.class);

    private final RetrievalPipeline retrievalPipeline;
    private final PromptAssembler promptAssembler;
    private final ChatClient chatClient;

    public AnswerService(RetrievalPipeline retrievalPipeline, PromptAssembler promptAssembler,
            ChatClient.Builder chatClientBuilder) {
        this.retrievalPipeline = retrievalPipeline;
        this.promptAssembler = promptAssembler;
        this.chatClient = chatClientBuilder.build();
    }

    public Flux<ChatEvent> answer(String question) {
        return Mono.fromCallable(() -> retrievalPipeline.retrieve(question))
                .subscribeOn(Schedulers.boundedElastic()) // JDBC + embedding call are blocking
                .flatMapMany(retrieval -> generate(question, retrieval))
                .onErrorResume(error -> {
                    // Details (SQL errors, API error bodies) stay in the log; the browser gets a generic message.
                    log.error("Answering failed for question: {}", question, error);
                    return Flux.just(new ChatEvent.Error(ANSWER_FAILED));
                });
    }

    private Flux<ChatEvent> generate(String question, RetrievalResult retrieval) {
        List<Document> documents = retrieval.documents();
        ChatEvent sources = ChatEvent.Sources.from(retrieval);
        long retrievalMillis = retrieval.trace().totalMillis();
        if (documents.isEmpty()) {
            return Flux.just(sources, new ChatEvent.Token(NO_SOURCES_ANSWER),
                    new ChatEvent.Done(null, null, retrievalMillis, 0));
        }

        AssembledPrompt prompt = promptAssembler.assemble(question, documents);
        AtomicReference<Usage> usage = new AtomicReference<>();
        AtomicLong started = new AtomicLong();
        // Message objects, not .system()/.user() strings: retrieved code samples are full of {braces}
        // that must never be mistaken for template variables.
        Flux<ChatEvent> tokens = chatClient
                .prompt(new Prompt(List.of(new SystemMessage(prompt.system()), new UserMessage(prompt.user()))))
                .stream()
                .chatResponse()
                .doOnSubscribe(subscription -> started.set(System.nanoTime()))
                .doOnNext(response -> captureUsage(response, usage))
                .mapNotNull(AnswerService::textOf)
                .<ChatEvent>map(ChatEvent.Token::new);
        Mono<ChatEvent> done = Mono.fromSupplier(() -> ChatEvent.Done.of(usage.get(), retrievalMillis,
                TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started.get())));

        return Flux.concat(Mono.just(sources), tokens, done);
    }

    private static String textOf(ChatResponse response) {
        Generation generation = response.getResult();
        if (generation == null || generation.getOutput() == null) {
            return null; // e.g. the final usage-only chunk
        }
        String text = generation.getOutput().getText();
        return text == null || text.isEmpty() ? null : text;
    }

    private static void captureUsage(ChatResponse response, AtomicReference<Usage> usage) {
        if (response.getMetadata() == null) {
            return;
        }
        Usage current = response.getMetadata().getUsage();
        if (current != null && current.getTotalTokens() != null && current.getTotalTokens() > 0) {
            usage.set(current);
        }
    }
}
