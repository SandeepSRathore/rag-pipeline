package com.learnings.rag.eval;

import static java.util.stream.Collectors.joining;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.RunInfo;

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

        md.append("## Summary\n\n| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |\n|---|---|---|---|---|---|\n");
        for (ConfigResult config : report.configs()) {
            RetrievalMetrics.Summary summary = config.summary();
            md.append("| ").append(config.name())
                    .append(" | ").append(decimal(summary.hitAt5()))
                    .append(" | ").append(decimal(summary.recallAt5()))
                    .append(" | ").append(decimal(summary.mrrAt10()))
                    .append(" | ").append(summary.p50Millis())
                    .append(" | ").append(summary.p95Millis()).append(" |\n");
        }

        for (ConfigResult config : report.configs()) {
            md.append("\n## ").append(config.name()).append(": per question\n\n")
                    .append("| Id | First relevant rank | hit@5 | recall@5 | Question |\n|---|---|---|---|---|\n");
            for (ItemResult item : config.items()) {
                Integer rank = item.score().firstRelevantRank();
                md.append("| ").append(item.id())
                        .append(" | ").append(rank == null ? "–" : rank)
                        .append(" | ").append(item.score().hitAt5() ? "✓" : "✗")
                        .append(" | ").append(decimal(item.score().recallAt5()))
                        .append(" | ").append(cell(item.question())).append(" |\n");
            }
            List<ItemResult> misses = config.items().stream()
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
        }
        return md.toString();
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
