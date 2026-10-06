package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

class ReciprocalRankFusionTest {

    private static Document doc(String id, String retriever) {
        return Document.builder().id(id).text("text " + id).metadata(Map.of("from", retriever)).score(0.5).build();
    }

    private static List<Document> vector() {
        return List.of(doc("a", "vector"), doc("b", "vector"), doc("c", "vector"));
    }

    private static List<Document> keyword() {
        return List.of(doc("c", "keyword"), doc("d", "keyword"));
    }

    @Test
    void aDocumentInBothRankingsAppearsOnceWithSummedScore() {
        List<Document> fused = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10);

        assertThat(fused).extracting(Document::getId).containsExactly("c", "a", "b", "d");
        assertThat(fused.getFirst().getScore()).isCloseTo(1.0 / 63 + 1.0 / 61, within(1e-12));
    }

    @Test
    void tiesKeepTheOrderDocumentsWereFirstSeenIn() {
        // b (vector rank 2) and d (keyword rank 2) both score 1/62; b was seen first.
        List<Document> fused = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10);

        assertThat(fused.get(2).getScore()).isEqualTo(fused.get(3).getScore());
        assertThat(fused).extracting(Document::getId).containsSubsequence("b", "d");
    }

    @Test
    void theLimitCapsTheFusedList() {
        assertThat(ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 2))
                .extracting(Document::getId).containsExactly("c", "a");
    }

    @Test
    void theFirstSeenCopyOfADocumentIsKept() {
        Document fusedC = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10).getFirst();

        assertThat(fusedC.getMetadata()).containsEntry("from", "vector");
        assertThat(fusedC.getText()).isEqualTo("text c");
    }

    @Test
    void emptyRankingsFuseToNothing() {
        assertThat(ReciprocalRankFusion.fuse(List.of(List.of(), List.of()), 60, 10)).isEmpty();
    }

    @Test
    void aSingleRankingKeepsItsOrder() {
        assertThat(ReciprocalRankFusion.fuse(List.of(vector()), ReciprocalRankFusion.DEFAULT_K, 10))
                .extracting(Document::getId).containsExactly("a", "b", "c");
    }
}
