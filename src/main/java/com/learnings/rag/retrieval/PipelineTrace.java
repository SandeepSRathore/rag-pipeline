package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/**
 * What each retrieval stage did and how long it took; feeds the done event and the sources' per-stage scores.
 * Stages can run in parallel (HYBRID runs vector and keyword search at once), so {@code totalMillis} is the
 * wall-clock time of the whole retrieval, not the sum of the stages.
 */
public record PipelineTrace(List<Stage> stages, long totalMillis) {

    /** For stages that ran one after another: the total is their sum. */
    public PipelineTrace(List<Stage> stages) {
        this(stages, stages.stream().mapToLong(Stage::elapsedMillis).sum());
    }

    /** The queries that were searched: those of the last stage that lists any (expand, else rewrite); else empty. */
    public List<String> queries() {
        for (int i = stages.size() - 1; i >= 0; i--) {
            if (!stages.get(i).queries().isEmpty()) {
                return stages.get(i).queries();
            }
        }
        return List.of();
    }

    /** @param queries for rewrite and expand stages: the queries they produced */
    public record Stage(String name, long elapsedMillis, List<Hit> hits, List<String> queries) {

        public Stage(String name, long elapsedMillis, List<Hit> hits) {
            this(name, elapsedMillis, hits, List.of());
        }
    }

    public record Hit(String id, String sourcePath, String breadcrumb, Double score) {

        static Hit of(Document document) {
            return new Hit(document.getId(),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                    document.getScore());
        }
    }
}
