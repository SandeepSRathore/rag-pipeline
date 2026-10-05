package com.learnings.rag.generation;

import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/** @param n the citation number the model uses ([n]) */
public record SourceRef(int n, String sourcePath, String title, String breadcrumb, Double score, String text) {

    static SourceRef of(int n, Document document) {
        var metadata = document.getMetadata();
        return new SourceRef(n,
                Objects.toString(metadata.get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(metadata.get(ChunkMetadata.TITLE), ""),
                Objects.toString(metadata.get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore(),
                document.getText());
    }
}
