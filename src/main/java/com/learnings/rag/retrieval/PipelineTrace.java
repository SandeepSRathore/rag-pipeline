package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/** What each retrieval stage did and how long it took; feeds the done event now and the debug panel in M7. */
public record PipelineTrace(List<Stage> stages) {

    public record Stage(String name, long elapsedMillis, List<Hit> hits) {
    }

    public record Hit(String id, String sourcePath, String breadcrumb, Double score) {

        static Hit of(Document document) {
            return new Hit(document.getId(),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                    document.getScore());
        }
    }

    public long totalMillis() {
        return stages.stream().mapToLong(Stage::elapsedMillis).sum();
    }
}
