package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

import com.learnings.rag.config.RagProperties;

class RetrievalPipelineTest {

    private final VectorRetriever vectorRetriever = mock(VectorRetriever.class);
    private final KeywordRetriever keywordRetriever = mock(KeywordRetriever.class);
    private final QueryRewriter queryRewriter = mock(QueryRewriter.class);
    private final LlmQueryExpander queryExpander = mock(LlmQueryExpander.class);
    private final RetrievalPipeline pipeline = new RetrievalPipeline(vectorRetriever, keywordRetriever, queryRewriter, queryExpander,
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

    private static RetrievalOptions vectorOnly() {
        return new RetrievalOptions(5, 0.0, RetrievalMode.VECTOR, 20);
    }

    @Test
    void rewriteSearchesTheRewrittenQuestion() {
        when(queryRewriter.rewrite("hey which index thing for pgvector?")).thenReturn("pgvector index types");
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call ->
                List.of(doc("for: " + ((Query) call.getArgument(0)).text())));

        RetrievalResult result = pipeline.retrieve("hey which index thing for pgvector?", vectorOnly().withRewrite(true));

        assertThat(result.documents()).extracting(Document::getId).containsExactly("for: pgvector index types");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("rewrite", "vector");
        assertThat(result.trace().queries()).containsExactly("pgvector index types");
    }

    @Test
    void variantsAreSearchedInParallelAndJoinedWithoutDuplicates() {
        when(queryExpander.expand(any(), eq(2))).thenReturn(
                List.of(new Query("original"), new Query("variant one"), new Query("variant two")));
        Map<String, List<Document>> rankings = Map.of(
                "original", List.of(doc("shared"), doc("a")),
                "variant one", List.of(doc("b"), doc("shared")),
                "variant two", List.of(doc("shared"), doc("c")));
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call ->
                rankings.get(((Query) call.getArgument(0)).text()));

        RetrievalResult result = pipeline.retrieve("original", vectorOnly().withQueryVariants(2));

        assertThat(result.documents()).extracting(Document::getId).startsWith("shared").doesNotHaveDuplicates()
                .containsExactlyInAnyOrder("shared", "a", "b", "c");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("expand", "q1:vector", "q2:vector", "q3:vector", "join");
        assertThat(result.trace().queries()).containsExactly("original", "variant one", "variant two");
    }

    @Test
    void aFailingVariantSearchFailsWithItsOwnException() {
        when(queryExpander.expand(any(), eq(2))).thenReturn(
                List.of(new Query("original"), new Query("variant one"), new Query("variant two")));
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call -> {
            if (((Query) call.getArgument(0)).text().equals("variant two")) {
                throw new IllegalStateException("embedding service unavailable");
            }
            return List.of(doc("a"));
        });

        assertThatThrownBy(() -> pipeline.retrieve("original", vectorOnly().withQueryVariants(2)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("embedding service unavailable");
    }

    @Test
    void anExpansionThatFellBackToTheOriginalSearchesOnceWithM4StageNames() {
        when(queryExpander.expand(any(), eq(3))).thenReturn(List.of(new Query("original")));
        when(vectorRetriever.retrieve(any(), any())).thenReturn(List.of(doc("a")));

        RetrievalResult result = pipeline.retrieve("original", vectorOnly().withQueryVariants(3));

        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("expand", "vector");
        assertThat(result.trace().queries()).containsExactly("original");
    }
}
