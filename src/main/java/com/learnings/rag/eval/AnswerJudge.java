package com.learnings.rag.eval;

import java.util.List;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.evaluation.FactCheckingEvaluator;
import org.springframework.ai.chat.evaluation.RelevancyEvaluator;
import org.springframework.ai.document.Document;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * LLM-as-judge checks of one answer with Spring AI's evaluators, on the utility model: a different model from the one
 * that wrote the answer, so no model grades itself. Each check is one call that must reply "yes"; Spring AI accepts
 * exactly "yes" in any case, so a decorated reply ("Yes.") is a fail. A failed call is an {@link Verdict#ERROR},
 * never a fail.
 */
@Component
public class AnswerJudge {

    private static final Logger log = LoggerFactory.getLogger(AnswerJudge.class);

    public enum Verdict {
        PASS, FAIL, ERROR
    }

    private final FactCheckingEvaluator factChecker;
    private final RelevancyEvaluator relevancy;

    public AnswerJudge(@Qualifier("utilityChatClient") ChatClient utilityChatClient) {
        this.factChecker = FactCheckingEvaluator.builder(utilityChatClient.mutate()).build();
        this.relevancy = RelevancyEvaluator.builder().chatClientBuilder(utilityChatClient.mutate()).build();
    }

    /** Faithfulness: is the answer supported by the sources it was given? */
    public Verdict faithful(String answer, List<String> sources) {
        return verdict("faithfulness", () -> factChecker.evaluate(new EvaluationRequest(documents(sources), answer)));
    }

    /** Relevancy: does the answer respond to the question, in line with the sources? */
    public Verdict relevant(String question, String answer, List<String> sources) {
        return verdict("relevancy",
                () -> relevancy.evaluate(new EvaluationRequest(question, documents(sources), answer)));
    }

    /** Correctness: is the golden reference answer supported by the answer, i.e. does the answer contain its facts? */
    public Verdict correct(String answer, String referenceAnswer) {
        return verdict("correctness",
                () -> factChecker.evaluate(new EvaluationRequest(List.of(new Document(answer)), referenceAnswer)));
    }

    private static List<Document> documents(List<String> sources) {
        return sources.stream().map(Document::new).toList();
    }

    private static Verdict verdict(String check, Supplier<EvaluationResponse> evaluation) {
        try {
            return evaluation.get().isPass() ? Verdict.PASS : Verdict.FAIL;
        }
        catch (RuntimeException e) {
            log.warn("The {} judge failed; counted as an error, not a fail", check, e);
            return Verdict.ERROR;
        }
    }
}
