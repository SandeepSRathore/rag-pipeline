package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/**
 * @param chunks the retrieved chunks, best first, as the chat would hand them to the model
 * @param trace every stage with its queries, ranked hits, scores and timing
 */
public record RetrieveResponse(List<Chunk> chunks, PipelineTrace trace) {

    /** @param score what the chunk was ranked by in the last stage (cosine, ts_rank, RRF or the 0–10 rating) */
    public record Chunk(int rank, String id, String sourcePath, String breadcrumb, Double score, String text) {
    }

    static RetrieveResponse of(RetrievalResult result) {
        List<Document> documents = result.documents();
        return new RetrieveResponse(IntStream.range(0, documents.size())
                .mapToObj(i -> {
                    Document document = documents.get(i);
                    return new Chunk(i + 1, document.getId(),
                            Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                            Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                            document.getScore(), document.getText());
                })
                .toList(), result.trace());
    }
}
