package com.learnings.rag.eval;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.GenerationReport.Group;
import com.learnings.rag.eval.GenerationReport.Item;
import com.learnings.rag.eval.GenerationReport.Outcome;
import com.learnings.rag.generation.AnswerService;
import com.learnings.rag.generation.ChatEvent;
import com.learnings.rag.generation.CitationValidator;
import com.learnings.rag.generation.SourceRef;
import com.learnings.rag.retrieval.RetrievalOptions;

import reactor.core.publisher.Flux;

/**
 * Sends every golden question through the chat path ({@link AnswerService}, chat defaults) and judges what comes back:
 * refusals are classified, answered questions get faithfulness, relevancy, correctness (answerable only) and a
 * citation check. One question at a time, in golden-set order.
 */
@Service
public class GenerationEvalRunner {

    private static final Logger log = LoggerFactory.getLogger(GenerationEvalRunner.class);

    private static final Duration ANSWER_TIMEOUT = Duration.ofMinutes(5);

    private final AnswerService answerService;
    private final AnswerJudge judge;
    private final RetrievalOptions retrieval;
    private final String answerModel;
    private final String judgeModel;

    public GenerationEvalRunner(AnswerService answerService, AnswerJudge judge, RagProperties properties,
            @Value("${spring.ai.openai.chat.options.model:unknown}") String answerModel) {
        this.answerService = answerService;
        this.judge = judge;
        this.retrieval = RetrievalOptions.from(properties);
        this.answerModel = answerModel;
        this.judgeModel = properties.retrieval().utilityModel();
    }

    public GenerationReport run(List<GoldenItem> items) {
        List<Item> results = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Item result = evaluate(items.get(i));
            results.add(result);
            log.info("Generation {} ({}/{}): {} · faithful {} · relevant {} · correct {} · citations {}", result.id(),
                    i + 1, items.size(), result.outcome(), result.faithful(), result.relevant(), result.correct(),
                    result.citations() == null ? null : result.citations().valid() ? "valid" : "invalid");
        }
        List<Long> generation = results.stream().filter(item -> item.outcome() == Outcome.ANSWERED)
                .map(Item::generationMillis).toList();
        return new GenerationReport(answerModel, judgeModel, retrieval,
                Group.of(results.stream().filter(Item::answerable).toList()),
                Group.of(results.stream().filter(item -> !item.answerable()).toList()),
                generation.isEmpty() ? 0 : RetrievalMetrics.percentile(generation, 0.50), List.copyOf(results));
    }

    private Item evaluate(GoldenItem item) {
        Answer answer = collect(item.question());
        Outcome outcome = answer.failed() || answer.text().isBlank() ? Outcome.FAILED
                : answer.sources().isEmpty() ? Outcome.REFUSED_BY_RETRIEVAL
                : AnswerService.isRefusal(answer.text()) ? Outcome.REFUSED_BY_MODEL
                : Outcome.ANSWERED;
        if (outcome != Outcome.ANSWERED) {
            return new Item(item.id(), item.question(), item.answerable(), outcome, answer.sources().size(),
                    answer.text(), null, null, null, null, answer.generationMillis());
        }
        List<String> sources = answer.sources().stream().map(SourceRef::text).toList();
        return new Item(item.id(), item.question(), item.answerable(), outcome, sources.size(), answer.text(),
                judge.faithful(answer.text(), sources),
                judge.relevant(item.question(), answer.text(), sources),
                item.answerable() ? judge.correct(answer.text(), item.referenceAnswer()) : null,
                CitationValidator.check(answer.text(), sources.size()), answer.generationMillis());
    }

    private record Answer(List<SourceRef> sources, String text, boolean failed, long generationMillis) {
    }

    /** The chat's events for one question, folded into one answer; a timeout or an exception is a failed answer. */
    private Answer collect(String question) {
        List<ChatEvent> events;
        try {
            Flux<ChatEvent> stream = answerService.answer(question);
            events = stream.collectList().block(ANSWER_TIMEOUT);
        }
        catch (RuntimeException e) {
            log.warn("Answering failed for question: {}", question, e);
            return new Answer(List.of(), "", true, 0);
        }
        List<SourceRef> sources = List.of();
        StringBuilder text = new StringBuilder();
        boolean failed = false;
        long generationMillis = 0;
        for (ChatEvent event : events == null ? List.<ChatEvent>of() : events) {
            switch (event) {
                case ChatEvent.Sources s -> sources = s.sources();
                case ChatEvent.Trace _ -> {
                }
                case ChatEvent.Token t -> text.append(t.text());
                case ChatEvent.Done d -> generationMillis = d.generationMillis();
                case ChatEvent.Error _ -> failed = true;
            }
        }
        return new Answer(sources, text.toString(), failed, generationMillis);
    }
}
