package com.learnings.rag.eval;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import com.learnings.rag.eval.AnswerJudge.Verdict;
import com.learnings.rag.generation.CitationValidator;
import com.learnings.rag.retrieval.RetrievalOptions;

/**
 * What the chat answered for every golden question, and how the answers were judged.
 *
 * @param retrieval the chat's retrieval settings the answers were generated with
 * @param p50GenerationMillis median answer-model time over the answered questions; 0 when none was answered
 */
public record GenerationReport(String answerModel, String judgeModel, RetrievalOptions retrieval, Group answerable,
        Group unanswerable, long p50GenerationMillis, List<Item> items) {

    /** How the chat handled a question. Only answered questions are judged: a judge marks any refusal unsupported. */
    public enum Outcome {
        ANSWERED, REFUSED_BY_RETRIEVAL, REFUSED_BY_MODEL, FAILED
    }

    /**
     * @param sources how many sources the answer model was given
     * @param faithful judge verdicts; null when not judged (not answered, or no reference for correctness)
     * @param citations null when not answered
     */
    public record Item(String id, String question, boolean answerable, Outcome outcome, int sources, String answer,
            Verdict faithful, Verdict relevant, Verdict correct, CitationValidator.Check citations,
            long generationMillis) {
    }

    /** @param judged verdicts that passed or failed; {@code errors} are judge failures, counted apart */
    public record Rate(int passed, int judged, int errors) {

        static Rate of(List<Item> answered, Function<Item, Verdict> verdict) {
            List<Verdict> verdicts = answered.stream().map(verdict).filter(Objects::nonNull).toList();
            int passed = (int) verdicts.stream().filter(v -> v == Verdict.PASS).count();
            int errors = (int) verdicts.stream().filter(v -> v == Verdict.ERROR).count();
            return new Rate(passed, verdicts.size() - errors, errors);
        }
    }

    public record Group(int questions, int answered, int refusedByRetrieval, int refusedByModel, int failed,
            Rate faithful, Rate relevant, Rate correct, Rate citationsValid) {

        static Group of(List<Item> items) {
            List<Item> answered = items.stream().filter(item -> item.outcome() == Outcome.ANSWERED).toList();
            int citationsValid = (int) answered.stream().filter(item -> item.citations().valid()).count();
            return new Group(items.size(), answered.size(), count(items, Outcome.REFUSED_BY_RETRIEVAL),
                    count(items, Outcome.REFUSED_BY_MODEL), count(items, Outcome.FAILED),
                    Rate.of(answered, Item::faithful), Rate.of(answered, Item::relevant),
                    Rate.of(answered, Item::correct), new Rate(citationsValid, answered.size(), 0));
        }

        private static int count(List<Item> items, Outcome outcome) {
            return (int) items.stream().filter(item -> item.outcome() == outcome).count();
        }
    }

    /** Judge calls that failed, over every check and both groups. */
    public int judgeErrors() {
        return List.of(answerable, unanswerable).stream()
                .mapToInt(group -> group.faithful().errors() + group.relevant().errors() + group.correct().errors())
                .sum();
    }
}
