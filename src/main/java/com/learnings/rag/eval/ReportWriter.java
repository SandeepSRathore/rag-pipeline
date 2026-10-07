package com.learnings.rag.eval;

import static java.util.stream.Collectors.joining;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.AnswerJudge.Verdict;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.MinScoreRow;
import com.learnings.rag.eval.EvalReport.Refusals;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.EvalReport.TagSummary;
import com.learnings.rag.eval.GenerationReport.Group;
import com.learnings.rag.eval.GenerationReport.Item;
import com.learnings.rag.eval.GenerationReport.Outcome;
import com.learnings.rag.eval.GenerationReport.Rate;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.generation.CitationValidator;
import com.learnings.rag.retrieval.RetrievalOptions;

import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

/** Writes an eval report as Markdown (for reading and diffing) and JSON (every number, for later analysis). */
@Component
public class ReportWriter {

    private static final DateTimeFormatter FILE_NAME =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH-mm-ss'Z'").withZone(ZoneOffset.UTC);

    private final JsonMapper json;

    public ReportWriter(JsonMapper json) {
        this.json = json;
    }

    /** Writes {@code <start time>.md} and {@code <start time>.json} into {@code directory}; returns the Markdown path. */
    public Path write(EvalReport report, Path directory) {
        String name = FILE_NAME.format(report.startedAt());
        try {
            Files.createDirectories(directory);
            Path markdown = directory.resolve(name + ".md");
            Files.writeString(markdown, markdown(report));
            Files.write(directory.resolve(name + ".json"),
                    json.writer().with(SerializationFeature.INDENT_OUTPUT).writeValueAsBytes(report));
            return markdown;
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static String markdown(EvalReport report) {
        RunInfo run = report.run();
        RagProperties.Chunking chunking = run.chunking();
        StringBuilder md = new StringBuilder()
                .append("# Retrieval eval: ").append(report.startedAt()).append("\n\n")
                .append("| | |\n|---|---|\n")
                .append("| Golden set | `").append(run.goldenSet()).append("` (").append(run.goldenItems())
                .append(" questions, sha256 `").append(run.goldenSetSha256(), 0, 12).append("…`) |\n")
                .append("| Index | ").append(run.index().corpusDocuments()).append(" corpus pages, ")
                .append(run.index().uploadedDocuments()).append(" uploads, ").append(run.index().chunks())
                .append(" chunks |\n")
                .append("| Embedding model | `").append(run.embeddingModel()).append("` |\n")
                .append("| Chunking | max ").append(chunking.maxTokens()).append(" · min ").append(chunking.minTokens())
                .append(" · overlap ").append(chunking.overlapTokens()).append(" tokens |\n\n");
        if (run.index().uploadedDocuments() > 0) {
            md.append("> **Warning:** the index contains ").append(run.index().uploadedDocuments())
                    .append(" uploaded document(s). They compete with corpus pages for every question, so these ")
                    .append("numbers are not comparable with a corpus-only run.\n\n");
        }

        md.append("## Summary\n\n| Config | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable "
                + "| p50 ms | p95 ms |\n|---|---|---|---|---|---|---|---|\n");
        for (ConfigResult config : report.configs()) {
            RetrievalMetrics.Summary summary = config.summary();
            md.append("| ").append(config.name())
                    .append(" | ").append(decimal(summary.hitAt5()))
                    .append(" | ").append(decimal(summary.recallAt5()))
                    .append(" | ").append(decimal(summary.mrrAt10()))
                    .append(" | ").append(refusedUnanswerable(config.refusals()))
                    .append(" | ").append(config.refusals().answerableRefused())
                    .append(" | ").append(summary.p50Millis())
                    .append(" | ").append(summary.p95Millis()).append(" |\n");
        }
        md.append("\nAnswerable questions: ").append(report.configs().getFirst().summary().items())
                .append("; hit@5, recall@5, MRR@10 and latency cover these. *Refused* = retrieval came back empty, ")
                .append("so chat answers \"I couldn't find…\" without calling the model.\n");

        md.append("\nQueries searched per question (average): ").append(report.configs().stream()
                .map(config -> {
                    double average = config.items().stream().mapToInt(item -> Math.max(1, item.queries().size()))
                            .average().orElse(1);
                    String text = config.name() + " " + String.format(Locale.ROOT, "%.1f", average);
                    if (config.options().queryVariants() > 0) {
                        long fellBack = config.items().stream().filter(item -> item.queries().size() <= 1).count();
                        text += " (expansion fell back on " + fellBack + ")";
                    }
                    return text;
                })
                .collect(joining(" · "))).append('\n');
        if (report.configs().stream().anyMatch(config -> config.options().rerank())) {
            md.append("\nRerank fell back to the fused order (questions): ").append(report.configs().stream()
                    .filter(config -> config.options().rerank())
                    .map(config -> config.name() + " "
                            + config.items().stream().filter(ItemResult::rerankFellBack).count())
                    .collect(joining(" · "))).append('\n');
        }
        if (report.configs().stream().anyMatch(config -> !config.byTag().isEmpty())) {
            md.append("\n## By tag\n\n| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |\n|---|---|---|---|---|---|\n");
            for (ConfigResult config : report.configs()) {
                for (TagSummary tag : config.byTag()) {
                    RetrievalMetrics.Summary summary = tag.summary();
                    md.append("| ").append(config.name())
                            .append(" | ").append(tag.tag())
                            .append(" | ").append(summary.items())
                            .append(" | ").append(decimal(summary.hitAt5()))
                            .append(" | ").append(decimal(summary.recallAt5()))
                            .append(" | ").append(decimal(summary.mrrAt10())).append(" |\n");
                }
            }
        }

        for (ConfigResult config : report.configs()) {
            if (config.minScoreSweep().isEmpty()) {
                continue;
            }
            md.append("\n## ").append(config.name()).append(": min-score sweep\n\n")
                    .append("Each row is what this run would have scored with `rag.retrieval.rerank.min-score` at that ")
                    .append("value: chunks rated below it are dropped, and a question left with no chunks is refused. ")
                    .append("Questions whose rerank failed keep their chunks.\n\n")
                    .append("| min-score | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable |\n")
                    .append("|---|---|---|---|---|---|\n");
            for (MinScoreRow row : config.minScoreSweep()) {
                md.append("| ").append(row.minScore())
                        .append(" | ").append(decimal(row.hitAt5()))
                        .append(" | ").append(decimal(row.recallAt5()))
                        .append(" | ").append(decimal(row.mrrAt10()))
                        .append(" | ").append(refusedUnanswerable(row.refusals()))
                        .append(" | ").append(row.refusals().answerableRefused()).append(" |\n");
            }
        }

        for (ConfigResult config : report.configs()) {
            md.append("\n## ").append(config.name()).append(": per question\n\n")
                    .append("| Id | First relevant rank | hit@5 | recall@5 | Question |\n|---|---|---|---|---|\n");
            for (ItemResult item : config.items().stream().filter(ItemResult::answerable).toList()) {
                Integer rank = item.score().firstRelevantRank();
                md.append("| ").append(item.id())
                        .append(" | ").append(rank == null ? "–" : rank)
                        .append(" | ").append(item.score().hitAt5() ? "✓" : "✗")
                        .append(" | ").append(decimal(item.score().recallAt5()))
                        .append(" | ").append(cell(item.question())).append(" |\n");
            }
            List<ItemResult> misses = config.items().stream().filter(ItemResult::answerable)
                    .filter(item -> item.score().firstRelevantRank() == null)
                    .toList();
            md.append("\n### Misses: no relevant chunk in the top ").append(RetrievalMetrics.MRR_K).append("\n\n");
            if (misses.isEmpty()) {
                md.append("None.\n");
            }
            for (ItemResult miss : misses) {
                md.append("- **").append(miss.id()).append("** ").append(oneLine(miss.question())).append('\n')
                        .append("  - expected: ").append(miss.expectedSources().stream()
                                .map(source -> label(source.sourcePath(), source.sectionPrefix()))
                                .collect(joining("; ")))
                        .append('\n')
                        .append("  - retrieved: ").append(miss.retrieved().stream().limit(3)
                                .map(chunk -> label(chunk.sourcePath(), chunk.breadcrumb()))
                                .collect(joining("; ")))
                        .append('\n');
            }

            List<ItemResult> unanswerable = config.items().stream().filter(item -> !item.answerable()).toList();
            if (!unanswerable.isEmpty()) {
                md.append("\n### Unanswerable: retrieval should come back empty\n\n")
                        .append("| Id | Refused | Top chunk | Score | Question |\n|---|---|---|---|---|\n");
                for (ItemResult item : unanswerable) {
                    RankedSource top = item.retrieved().isEmpty() ? null : item.retrieved().getFirst();
                    md.append("| ").append(item.id())
                            .append(" | ").append(top == null ? "✓" : "✗")
                            .append(" | ").append(top == null ? "–" : cell(label(top.sourcePath(), top.breadcrumb())))
                            .append(" | ").append(top == null || top.score() == null ? "–" : decimal(top.score()))
                            .append(" | ").append(cell(item.question())).append(" |\n");
                }
            }
        }
        if (report.generation() != null) {
            generation(md, report.generation());
        }
        return md.toString();
    }

    private static String refusedUnanswerable(Refusals refusals) {
        return refusals.unanswerable() == 0 ? "–" : refusals.unanswerableRefused() + "/" + refusals.unanswerable();
    }

    private static void generation(StringBuilder md, GenerationReport generation) {
        RetrievalOptions retrieval = generation.retrieval();
        md.append("\n## Generation\n\n")
                .append("Answer model `").append(generation.answerModel()).append("`, judge `")
                .append(generation.judgeModel()).append("`. Every question went through the chat path with its defaults: ")
                .append(retrieval.mode().name().toLowerCase(Locale.ROOT)).append(" retrieval, top ")
                .append(retrieval.topK())
                .append(retrieval.rerank() ? ", rerank on, min-score " + number(retrieval.minScore()) : ", rerank off")
                .append(".\n\n")
                .append("| Questions | Count | Answered | Refused by retrieval | Refused by model | Failed | Faithful ")
                .append("| Relevant | Correct | Citations valid |\n|---|---|---|---|---|---|---|---|---|---|\n");
        groupRow(md, "answerable", generation.answerable());
        groupRow(md, "unanswerable", generation.unanswerable());
        md.append("\nFaithful, Relevant and Correct are judge verdicts on the answered questions; Correct needs a reference ")
                .append("answer, so it covers answerable questions only. p50 generation: ")
                .append(generation.p50GenerationMillis()).append(" ms. Judge errors (not counted as fails): ")
                .append(generation.judgeErrors()).append(".\n");

        md.append("\n### Generation: per question\n\n")
                .append("| Id | Outcome | Sources | Faithful | Relevant | Correct | Citations | Question |\n")
                .append("|---|---|---|---|---|---|---|---|\n");
        for (Item item : generation.items()) {
            md.append("| ").append(item.id())
                    .append(" | ").append(outcome(item.outcome()))
                    .append(" | ").append(item.sources())
                    .append(" | ").append(verdict(item.faithful()))
                    .append(" | ").append(verdict(item.relevant()))
                    .append(" | ").append(verdict(item.correct()))
                    .append(" | ").append(item.citations() == null ? "–" : item.citations().valid() ? "✓" : "✗")
                    .append(" | ").append(cell(item.question())).append(" |\n");
        }

        md.append("\n### Answers to review\n\n");
        List<Item> toReview = generation.items().stream().filter(item -> !reviewReasons(item).isEmpty()).toList();
        if (toReview.isEmpty()) {
            md.append("None.\n");
        }
        for (Item item : toReview) {
            String answer = oneLine(item.answer());
            md.append("- **").append(item.id()).append("** ").append(oneLine(item.question())).append(" — ")
                    .append(String.join("; ", reviewReasons(item))).append('\n')
                    .append("  > ").append(answer.length() > 300 ? answer.substring(0, 300) + "…" : answer).append('\n');
        }
    }

    private static void groupRow(StringBuilder md, String name, Group group) {
        md.append("| ").append(name)
                .append(" | ").append(group.questions())
                .append(" | ").append(group.answered())
                .append(" | ").append(group.refusedByRetrieval())
                .append(" | ").append(group.refusedByModel())
                .append(" | ").append(group.failed())
                .append(" | ").append(rate(group.faithful()))
                .append(" | ").append(rate(group.relevant()))
                .append(" | ").append(rate(group.correct()))
                .append(" | ").append(rate(group.citationsValid())).append(" |\n");
    }

    /** Why an answer deserves a look: a wrong refusal, an answered unanswerable question, a failed check. */
    private static List<String> reviewReasons(Item item) {
        List<String> reasons = new ArrayList<>();
        if (item.outcome() == Outcome.FAILED) {
            reasons.add("generation failed");
        }
        else if (item.outcome() != Outcome.ANSWERED) {
            if (item.answerable()) {
                reasons.add("refused an answerable question (" + outcome(item.outcome()) + ")");
            }
            // A refusal is never judged: one that also cites sources may have answered anyway (e.g. with another
            // product's settings), so it is shown rather than counted silently as a refusal.
            if (item.outcome() == Outcome.REFUSED_BY_MODEL
                    && !CitationValidator.check(item.answer(), item.sources()).cited().isEmpty()) {
                reasons.add("refusal that also cites sources");
            }
        }
        else {
            if (!item.answerable()) {
                reasons.add("answered an unanswerable question");
            }
            check(reasons, "faithfulness", item.faithful());
            check(reasons, "relevancy", item.relevant());
            check(reasons, "correctness", item.correct());
            if (!item.citations().valid()) {
                reasons.add(item.citations().cited().isEmpty() ? "no citations"
                        : "citations out of range: " + item.citations().outOfRange().stream().map(String::valueOf)
                                .collect(joining(", ")));
            }
        }
        return reasons;
    }

    private static void check(List<String> reasons, String name, Verdict verdict) {
        if (verdict == Verdict.FAIL) {
            reasons.add(name + " failed");
        }
        else if (verdict == Verdict.ERROR) {
            reasons.add(name + " judge error");
        }
    }

    private static String outcome(Outcome outcome) {
        return switch (outcome) {
            case ANSWERED -> "answered";
            case REFUSED_BY_RETRIEVAL -> "refused by retrieval";
            case REFUSED_BY_MODEL -> "refused by model";
            case FAILED -> "failed";
        };
    }

    private static String verdict(Verdict verdict) {
        if (verdict == null) {
            return "–";
        }
        return switch (verdict) {
            case PASS -> "✓";
            case FAIL -> "✗";
            case ERROR -> "error";
        };
    }

    private static String rate(Rate rate) {
        return rate.judged() == 0 ? "–" : rate.passed() + "/" + rate.judged();
    }

    private static String number(double value) {
        return value == Math.rint(value) ? String.valueOf((long) value) : String.valueOf(value);
    }

    private static String decimal(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    private static String oneLine(String text) {
        return text.replaceAll("\\s+", " ").strip();
    }

    private static String cell(String text) {
        return oneLine(text).replace("|", "\\|");
    }

    private static String label(String sourcePath, String section) {
        return "`" + sourcePath + "`" + (section.isEmpty() ? "" : " › " + section);
    }
}
