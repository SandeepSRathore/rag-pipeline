package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import com.learnings.rag.config.RagProperties;

class RetrievalPipelineTest {

    private final VectorRetriever vectorRetriever = mock(VectorRetriever.class);
    private final KeywordRetriever keywordRetriever = mock(KeywordRetriever.class);
    private final RetrievalPipeline pipeline = new RetrievalPipeline(vectorRetriever, keywordRetriever,
            new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                    new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, "gpt-4.1-mini")));

    private static Document doc(String id) {
        return Document.builder().id(id).text("text " + id).score(0.5).build();
    }

    @Test
    void aFailingRetrieverFailsTheHybridSearchWithItsOwnException() {
        when(vectorRetriever.retrieve(any(), any())).thenThrow(new IllegalStateException("embedding service unavailable"));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of(doc("k1")));

        assertThatThrownBy(() -> pipeline.retrieve("question"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("embedding service unavailable");
    }

    @Test
    void hybridStillAnswersWhenKeywordSearchFindsNothing() {
        when(vectorRetriever.retrieve(any(), any())).thenReturn(List.of(doc("v1"), doc("v2")));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of());

        RetrievalResult result = pipeline.retrieve("how do I do it?");

        assertThat(result.documents()).extracting(Document::getId).containsExactly("v1", "v2");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("vector", "keyword", "fusion");
    }

    @Test
    void hybridAsksEachRetrieverForTheCandidateCountAndReturnsTopK() {
        when(vectorRetriever.retrieve(any(), any())).thenReturn(
                List.of(doc("a"), doc("b"), doc("c"), doc("d"), doc("e"), doc("f")));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of(doc("f"), doc("g")));

        RetrievalResult result = pipeline.retrieve("question");

        assertThat(result.documents()).hasSize(5).extracting(Document::getId).doesNotHaveDuplicates();
        assertThat(result.documents().getFirst().getId()).isEqualTo("f"); // in both rankings
    }
}
