package com.learnings.rag.eval;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.ingest.SourceDocumentRepository;
import com.learnings.rag.retrieval.RetrievalPipeline;
import com.learnings.rag.retrieval.RetrievalResult;

/** Runs every golden question through the retrieval pipeline, once per configuration, and scores the results. */
@Service
public class EvalRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalRunner.class);

    private final RetrievalPipeline pipeline;
    private final ChunkCatalog catalog;
    private final SourceDocumentRepository documents;
    private final RagProperties properties;

    public EvalRunner(RetrievalPipeline pipeline, ChunkCatalog catalog, SourceDocumentRepository documents,
            RagProperties properties) {
        this.pipeline = pipeline;
        this.catalog = catalog;
        this.documents = documents;
        this.properties = properties;
    }

    public EvalReport run(GoldenSet goldenSet, List<EvalConfig> configs) {
        IndexStats index = catalog.stats();
        if (index.chunks() == 0) {
            throw new IllegalStateException("The index has no chunks for embedding model "
                    + properties.embeddingModel() + ". Run scripts/fetch-corpus.sh and POST /api/ingest/corpus first.");
        }
        List<String> unindexed = unindexedSources(goldenSet);
        if (!unindexed.isEmpty()) {
            throw new IllegalStateException("Expected sources that are not in the index: " + unindexed
                    + ". Fix the paths in " + goldenSet.path() + ", or ingest the corpus.");
        }

        Instant startedAt = Instant.now();
        List<ConfigResult> results = configs.stream().map(config -> run(goldenSet.items(), config)).toList();
        RunInfo info = new RunInfo(goldenSet.path().toString(), goldenSet.sha256(), goldenSet.items().size(),
                properties.embeddingModel(), properties.chunking(), index);
        return new EvalReport(startedAt, info, results);
    }

    /** Expected source paths missing from the index: a renamed page or a typo would otherwise silently score 0. */
    public List<String> unindexedSources(GoldenSet goldenSet) {
        return goldenSet.items().stream()
                .flatMap(item -> item.expectedSources().stream())
                .map(ExpectedSource::sourcePath)
                .distinct()
                .filter(path -> documents.findBySourcePath(path).isEmpty())
                .sorted()
                .toList();
    }

    private ConfigResult run(List<GoldenItem> items, EvalConfig config) {
        pipeline.retrieve(items.getFirst().question(), config.options()); // warm-up: pool, HTTP client, JIT
        List<ItemResult> results = new ArrayList<>(items.size());
        for (GoldenItem item : items) {
            long start = System.nanoTime();
            RetrievalResult retrieval = pipeline.retrieve(item.question(), config.options());
            long millis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            List<RankedSource> ranked = retrieval.documents().stream().map(EvalRunner::ranked).toList();
            results.add(new ItemResult(item.id(), item.question(), item.expectedSources(),
                    RetrievalMetrics.score(item.expectedSources(), ranked), millis, ranked));
        }
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(
                results.stream().map(ItemResult::score).toList(),
                results.stream().map(ItemResult::latencyMillis).toList());
        log.info("{}: hit@5 {} · recall@5 {} · MRR@10 {} · p50 {} ms · p95 {} ms", config.name(),
                summary.hitAt5(), summary.recallAt5(), summary.mrrAt10(), summary.p50Millis(), summary.p95Millis());
        return new ConfigResult(config.name(), config.options(), summary, results);
    }

    private static RankedSource ranked(Document document) {
        return new RankedSource(
                Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore());
    }
}
