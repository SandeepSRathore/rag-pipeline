package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.MinScoreRow;
import com.learnings.rag.eval.EvalReport.Refusals;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.RetrievalMetrics.ItemScore;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.retrieval.RetrievalMode;
import com.learnings.rag.retrieval.RetrievalOptions;

import tools.jackson.databind.json.JsonMapper;

class ReportWriterTest {

    @TempDir
    Path dir;

    private static EvalReport report(int uploads) {
        ItemResult hit = new ItemResult("q01", "Which index | type builds a graph?",
                List.of(new ExpectedSource("api/vectordbs/pgvector.adoc", "Indexes")),
                new ItemScore(1, true, 1.0, 1.0), 120,
                List.of(new RankedSource("api/vectordbs/pgvector.adoc", "Indexes › HNSW", 0.61)));
        ItemResult miss = new ItemResult("q02", "How do I\nstream tokens?",
                List.of(new ExpectedSource("api/chatclient.adoc", "Streaming")),
                new ItemScore(null, false, 0.0, 0.0), 80,
                List.of(new RankedSource("upgrade-notes.adoc", "Upgrading to 1.0.0-M8", 0.2)));
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(List.of(hit.score(), miss.score()),
                List.of(120L, 80L));
        RunInfo run = new RunInfo("eval/golden-set.json", "a".repeat(64), 2, "text-embedding-3-small",
                new RagProperties.Chunking(500, 50, 60), new IndexStats(52, uploads, 1106));
        List<EvalReport.TagSummary> byTag = List.of(
                new EvalReport.TagSummary("identifier", RetrievalMetrics.summarize(List.of(miss.score()), List.of(80L))),
                new EvalReport.TagSummary("untagged", RetrievalMetrics.summarize(List.of(hit.score()), List.of(120L))));
        return new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), run,
                List.of(new ConfigResult("vector", new RetrievalOptions(10, 0.0, RetrievalMode.VECTOR, 20), summary, byTag, List.of(hit, miss))));
    }

    @Test
    void writesMarkdownAndJsonNamedAfterTheStartTime() {
        Path markdown = new ReportWriter(JsonMapper.builder().build()).write(report(0), dir.resolve("reports"));

        assertThat(markdown).hasFileName("2026-10-06T09-30-00Z.md").exists();
        assertThat(dir.resolve("reports/2026-10-06T09-30-00Z.json")).content()
                .contains("\"hitAt5\"", "text-embedding-3-small", "Upgrading to 1.0.0-M8");
    }

    @Test
    void summaryRowShowsMetricsToThreeDecimalsAndLatencyPercentiles() {
        assertThat(ReportWriter.markdown(report(0))).contains("| vector | 0.500 | 0.500 | 0.500 | – | 0 | 80 | 120 |");
    }

    @Test
    void missesListWhatWasExpectedAndWhatCameBack() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "- **q02** How do I stream tokens?",
                "expected: `api/chatclient.adoc` › Streaming",
                "retrieved: `upgrade-notes.adoc` › Upgrading to 1.0.0-M8");
    }

    @Test
    void questionsCannotBreakTheTable() {
        assertThat(ReportWriter.markdown(report(0))).contains("| Which index \\| type builds a graph? |");
    }

    @Test
    void warnsWhenUploadsShareTheIndex() {
        assertThat(ReportWriter.markdown(report(2))).contains("2 uploaded document(s)");
        assertThat(ReportWriter.markdown(report(0))).doesNotContain("uploaded document");
    }

    @Test
    void headerRecordsWhatWasMeasured() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "`eval/golden-set.json` (2 questions, sha256 `aaaaaaaaaaaa…`)",
                "52 corpus pages, 0 uploads, 1106 chunks",
                "`text-embedding-3-small`",
                "max 500 · min 50 · overlap 60 tokens");
    }

    @Test
    void byTagTableShowsEachTagPerConfig() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "## By tag",
                "| vector | identifier | 1 | 0.000 | 0.000 | 0.000 |",
                "| vector | untagged | 1 | 1.000 | 1.000 | 1.000 |");
    }

    @Test
    void reportsHowManyQueriesEachConfigSearchedAndHowOftenExpansionFellBack() {
        ExpectedSource source = new ExpectedSource("a.adoc", "");
        ItemResult expanded = new ItemResult("q01", "Question?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 900,
                List.of(), List.of("Question?", "v1", "v2", "v3"), false);
        ItemResult fellBack = new ItemResult("q02", "Question 2?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 400,
                List.of(), List.of("Question 2?"), false);
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(List.of(expanded.score(), fellBack.score()),
                List.of(900L, 400L));
        EvalReport report = new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), report(0).run(), List.of(
                new ConfigResult("hybrid+multiquery", new RetrievalOptions(10, 0.0, RetrievalMode.HYBRID, 20).withQueryVariants(3),
                        summary, List.of(), List.of(expanded, fellBack))));

        assertThat(ReportWriter.markdown(report))
                .contains("Queries searched per question (average): hybrid+multiquery 2.5 (expansion fell back on 1)");
    }

    @Test
    void unanswerableQuestionsAreReportedByRefusalAndLeftOutOfTheMetrics() {
        ExpectedSource source = new ExpectedSource("a.adoc", "");
        ItemResult answered = new ItemResult("q01", "Question?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 100,
                List.of(new RankedSource("a.adoc", "", 0.5)));
        ItemResult refused = new ItemResult("u01", "How long should I proof sourdough?", List.of(), null, 90, List.of());
        ItemResult answeredAnyway = new ItemResult("u02", "Which properties configure Pinecone?", List.of(), null, 95,
                List.of(new RankedSource("upgrade-notes.adoc", "Pinecone", 0.2)));
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(List.of(answered.score()), List.of(100L));
        EvalReport report = new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), report(0).run(), List.of(
                new ConfigResult("hybrid", new RetrievalOptions(10, 0.0, RetrievalMode.HYBRID, 20), summary, List.of(),
                        List.of(answered, refused, answeredAnyway))));

        String markdown = ReportWriter.markdown(report);

        assertThat(markdown).contains(
                "| hybrid | 1.000 | 1.000 | 1.000 | 1/2 | 0 | 100 | 100 |",
                "Answerable questions: 1;",
                "### Unanswerable: retrieval should come back empty",
                "| u01 | ✓ | – | – | How long should I proof sourdough? |",
                "| u02 | ✗ | `upgrade-notes.adoc` › Pinecone | 0.200 | Which properties configure Pinecone? |");
        assertThat(markdown).doesNotContain("| u01 | –", "- **u02**"); // not in the rank table or the misses
    }

    @Test
    void rerankConfigsShowTheMinScoreSweepAndHowOftenRerankFellBack() {
        ExpectedSource source = new ExpectedSource("a.adoc", "");
        ItemResult reranked = new ItemResult("q01", "Question?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 900,
                List.of(new RankedSource("a.adoc", "", 8.0)), List.of("Question?"), false);
        ItemResult fellBack = new ItemResult("q02", "Question 2?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 400,
                List.of(new RankedSource("a.adoc", "", 0.03)), List.of("Question 2?"), true);
        List<ItemResult> items = List.of(reranked, fellBack);
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(items.stream().map(ItemResult::score).toList(),
                List.of(900L, 400L));
        List<MinScoreRow> sweep = List.of(new MinScoreRow(0, 1.0, 1.0, 1.0, new Refusals(0, 0, 0)),
                new MinScoreRow(9, 0.5, 0.5, 0.5, new Refusals(0, 0, 1)));
        EvalReport report = new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), report(0).run(), List.of(
                new ConfigResult("hybrid+rerank", new RetrievalOptions(10, 0.0, RetrievalMode.HYBRID, 20).withRerank(true),
                        summary, List.of(), Refusals.of(items), sweep, items)));

        assertThat(ReportWriter.markdown(report)).contains(
                "Rerank fell back to the fused order (questions): hybrid+rerank 1",
                "## hybrid+rerank: min-score sweep",
                "| 0 | 1.000 | 1.000 | 1.000 | – | 0 |",
                "| 9 | 0.500 | 0.500 | 0.500 | – | 1 |");
    }

    @Test
    void configsWithoutRerankShowNoSweep() {
        assertThat(ReportWriter.markdown(report(0))).doesNotContain("min-score sweep", "Rerank fell back");
    }
}
