package com.learnings.rag.eval;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.MinScoreRow;
import com.learnings.rag.eval.EvalReport.Refusals;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.EvalReport.TagSummary;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.retrieval.RetrievalPipeline;
import com.learnings.rag.retrieval.RetrievalResult;

/** Runs every golden question through the retrieval pipeline, once per configuration, and scores the results. */
@Service
public class EvalRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalRunner.class);

    static final String UNTAGGED = "untagged";

    private final RetrievalPipeline pipeline;
    private final ChunkCatalog catalog;
    private final RagProperties properties;

    public EvalRunner(RetrievalPipeline pipeline, ChunkCatalog catalog, RagProperties properties) {
        this.pipeline = pipeline;
        this.catalog = catalog;
        this.properties = properties;
    }

    public EvalReport run(GoldenSet goldenSet, List<EvalConfig> configs) {
        IndexStats index = catalog.stats();
        if (index.chunks() == 0) {
            throw new IllegalStateException("The index has no chunks for embedding model "
                    + properties.embeddingModel() + ". Run scripts/fetch-corpus.sh and POST /api/ingest/corpus first.");
        }
        List<String> unmatchable = unmatchableSources(goldenSet);
        if (!unmatchable.isEmpty()) {
            throw new IllegalStateException("Expected sources that match no indexed chunk (wrong page or section, "
                    + "or a page without chunks): " + unmatchable + ". Fix them in " + goldenSet.path()
                    + ", or ingest the corpus.");
        }

        Instant startedAt = Instant.now();
        List<ConfigResult> results = configs.stream().map(config -> run(goldenSet.items(), config)).toList();
        RunInfo info = new RunInfo(goldenSet.path().toString(), goldenSet.sha256(), goldenSet.items().size(),
                properties.embeddingModel(), properties.chunking(), index);
        return new EvalReport(startedAt, info, results);
    }

    /**
     * Expected sources that no chunk of the current embedding model can match, as {@code "id: path › section"}: a
     * page that isn't indexed or has no chunks, or a renamed or mistyped section ({@code "Indexes > HNSW"}, a
     * trailing space). Scoring them would report a label mistake as a retrieval miss.
     */
    public List<String> unmatchableSources(GoldenSet goldenSet) {
        Map<String, Set<String>> breadcrumbs = catalog.breadcrumbsByPage();
        List<String> unmatchable = new ArrayList<>();
        for (GoldenItem item : goldenSet.items()) {
            for (ExpectedSource source : item.expectedSources()) {
                boolean matchesAChunk = breadcrumbs.getOrDefault(source.sourcePath(), Set.of()).stream()
                        .anyMatch(breadcrumb -> source.matches(source.sourcePath(), breadcrumb));
                if (!matchesAChunk) {
                    unmatchable.add(item.id() + ": " + source.sourcePath()
                            + (source.sectionPrefix().isEmpty() ? "" : " › " + source.sectionPrefix()));
                }
            }
        }
        return unmatchable;
    }

    private ConfigResult run(List<GoldenItem> items, EvalConfig config) {
        pipeline.retrieve(items.getFirst().question(), config.options()); // warm-up: pool, HTTP client, JIT
        List<ItemResult> results = new ArrayList<>(items.size());
        for (GoldenItem item : items) {
            long start = System.nanoTime();
            RetrievalResult retrieval = pipeline.retrieve(item.question(), config.options());
            long millis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            List<RankedSource> ranked = retrieval.documents().stream().map(EvalRunner::ranked).toList();
            List<String> queries = retrieval.trace().queries().isEmpty() ? List.of(item.question())
                    : retrieval.trace().queries();
            // An unanswerable question has nothing to rank against: it is judged by whether retrieval came back empty.
            RetrievalMetrics.ItemScore score = item.answerable()
                    ? RetrievalMetrics.score(item.expectedSources(), ranked) : null;
            results.add(new ItemResult(item.id(), item.question(), item.expectedSources(), score, millis, ranked,
                    queries, retrieval.trace().rerankFellBack()));
        }
        List<GoldenItem> answerableItems = new ArrayList<>();
        List<ItemResult> answerable = new ArrayList<>();
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i).answerable()) {
                answerableItems.add(items.get(i));
                answerable.add(results.get(i));
            }
        }
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(
                answerable.stream().map(ItemResult::score).toList(),
                answerable.stream().map(ItemResult::latencyMillis).toList());
        Refusals refusals = Refusals.of(results);
        log.info("{}: hit@5 {} · recall@5 {} · MRR@10 {} · p50 {} ms · p95 {} ms · refused {}/{} unanswerable, {} answerable",
                config.name(), summary.hitAt5(), summary.recallAt5(), summary.mrrAt10(), summary.p50Millis(),
                summary.p95Millis(), refusals.unanswerableRefused(), refusals.unanswerable(),
                refusals.answerableRefused());
        // The sweep needs every rating, so only a run at minimum 0 can be swept.
        List<MinScoreRow> sweep = config.options().rerank() && config.options().minScore() == 0
                ? MinScoreSweep.sweep(results) : List.of();
        return new ConfigResult(config.name(), config.options(), summary,
                summarizeByTag(answerableItems, answerable), refusals, sweep, results);
    }

    /** One summary per tag, plus "untagged" for the rest, in tag order; empty when no item is tagged. */
    static List<TagSummary> summarizeByTag(List<GoldenItem> items, List<ItemResult> results) {
        if (items.stream().allMatch(item -> item.tags().isEmpty())) {
            return List.of();
        }
        Map<String, List<ItemResult>> groups = new TreeMap<>();
        for (int i = 0; i < items.size(); i++) {
            List<String> tags = items.get(i).tags().isEmpty() ? List.of(UNTAGGED) : items.get(i).tags();
            for (String tag : tags) {
                groups.computeIfAbsent(tag, key -> new ArrayList<>()).add(results.get(i));
            }
        }
        return groups.entrySet().stream()
                .map(group -> new TagSummary(group.getKey(), RetrievalMetrics.summarize(
                        group.getValue().stream().map(ItemResult::score).toList(),
                        group.getValue().stream().map(ItemResult::latencyMillis).toList())))
                .toList();
    }

    private static RankedSource ranked(Document document) {
        return new RankedSource(
                Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore());
    }
}
