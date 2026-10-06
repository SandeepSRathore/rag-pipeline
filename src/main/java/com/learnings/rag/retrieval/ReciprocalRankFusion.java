package com.learnings.rag.retrieval;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;

/**
 * Reciprocal rank fusion (Cormack, Clarke and Büttcher, 2009): {@code score(d) = Σ 1 / (k + rank_i(d))} over the
 * rankings that contain d. Only ranks count, so retrievers with incomparable scores (cosine similarity and
 * {@code ts_rank_cd}) combine without any normalization; {@code k} damps the influence of the very top ranks.
 */
public final class ReciprocalRankFusion {

    /** The constant from the original paper; the spec fixes it at 60. */
    public static final int DEFAULT_K = 60;

    private ReciprocalRankFusion() {
    }

    /**
     * Fuses rankings (each best first) by document id. Returns at most {@code limit} documents, best first, each
     * scored with its fused score; the first-seen copy of a document is kept, and ties keep first-seen order.
     */
    public static List<Document> fuse(List<List<Document>> rankings, int k, int limit) {
        Map<String, Double> scores = new LinkedHashMap<>();
        Map<String, Document> firstSeen = new HashMap<>();
        for (List<Document> ranking : rankings) {
            for (int i = 0; i < ranking.size(); i++) {
                Document document = ranking.get(i);
                scores.merge(document.getId(), 1.0 / (k + i + 1), Double::sum);
                firstSeen.putIfAbsent(document.getId(), document);
            }
        }
        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed()) // stable: ties keep first-seen order
                .limit(limit)
                .map(entry -> {
                    Document document = firstSeen.get(entry.getKey());
                    return Document.builder()
                            .id(document.getId())
                            .text(document.getText())
                            .metadata(document.getMetadata())
                            .score(entry.getValue())
                            .build();
                })
                .toList();
    }
}
