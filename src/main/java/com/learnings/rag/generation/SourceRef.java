package com.learnings.rag.generation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.retrieval.PipelineTrace;

/**
 * @param n the citation number the model uses ([n])
 * @param score what the chunk was ranked by: cosine similarity (vector), ts_rank_cd (keyword) or the fused RRF score
 * @param scores the chunk's score in each retrieval stage that returned it, in stage order,
 *        e.g. {vector=0.61, keyword=0.08, fusion=0.0325}
 */
public record SourceRef(int n, String sourcePath, String title, String breadcrumb, Double score, String text,
        Map<String, Double> scores) {

    public SourceRef(int n, String sourcePath, String title, String breadcrumb, Double score, String text) {
        this(n, sourcePath, title, breadcrumb, score, text, Map.of());
    }

    static SourceRef of(int n, Document document, PipelineTrace trace) {
        Map<String, Double> scores = new LinkedHashMap<>();
        for (PipelineTrace.Stage stage : trace.stages()) {
            stage.hits().stream()
                    .filter(hit -> hit.id().equals(document.getId()) && hit.score() != null)
                    .findFirst()
                    .ifPresent(hit -> scores.put(stage.name(), hit.score()));
        }
        var metadata = document.getMetadata();
        return new SourceRef(n,
                Objects.toString(metadata.get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(metadata.get(ChunkMetadata.TITLE), ""),
                Objects.toString(metadata.get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore(),
                document.getText(),
                Collections.unmodifiableMap(scores));
    }
}
