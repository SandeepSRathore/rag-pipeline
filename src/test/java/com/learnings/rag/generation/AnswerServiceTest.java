package com.learnings.rag.generation;

import static com.learnings.rag.ingest.ChunkMetadata.BREADCRUMB;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_PATH;
import static com.learnings.rag.ingest.ChunkMetadata.TITLE;
import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.core.io.ClassPathResource;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.retrieval.PipelineTrace;
import com.learnings.rag.retrieval.RetrievalPipeline;
import com.learnings.rag.retrieval.RetrievalResult;

import reactor.test.StepVerifier;

class AnswerServiceTest {

    private final RetrievalPipeline pipeline = mock(RetrievalPipeline.class);
    private final PromptAssembler assembler = new PromptAssembler(new ClassPathResource("prompts/answer-system.st"));

    private AnswerService service(StubChatModel model) {
        return new AnswerService(pipeline, assembler, ChatClient.builder(model));
    }

    private static RetrievalResult oneChunk() {
        Document chunk = Document.builder()
                .text("PGvector › Indexes\n\nHNSW is the default.")
                .score(0.82)
                .metadata(Map.<String, Object>of(SOURCE_PATH, "pgvector.adoc", TITLE, "PGvector", BREADCRUMB, "Indexes"))
                .build();
        return new RetrievalResult(List.of(chunk), new PipelineTrace(List.of()));
    }

    @Test
    void streamsSourcesThenTokensThenUsage() {
        when(pipeline.retrieve("How do I enable HNSW?")).thenReturn(oneChunk());
        StubChatModel model = new StubChatModel("HNSW is ", "the default [1].");

        StepVerifier.create(service(model).answer("How do I enable HNSW?"))
                .assertNext(event -> assertThat(event).isInstanceOfSatisfying(ChatEvent.Sources.class,
                        sources -> assertThat(sources.sources()).singleElement().satisfies(ref -> {
                            assertThat(ref.n()).isEqualTo(1);
                            assertThat(ref.sourcePath()).isEqualTo("pgvector.adoc");
                            assertThat(ref.breadcrumb()).isEqualTo("Indexes");
                            assertThat(ref.score()).isEqualTo(0.82);
                        })))
                .expectNext(new ChatEvent.Token("HNSW is "), new ChatEvent.Token("the default [1]."))
                .assertNext(event -> assertThat(event).isInstanceOfSatisfying(ChatEvent.Done.class, done -> {
                    assertThat(done.promptTokens()).isEqualTo(120);
                    assertThat(done.completionTokens()).isEqualTo(7);
                }))
                .verifyComplete();

        assertThat(model.prompts()).singleElement()
                .satisfies(prompt -> assertThat(prompt.getContents())
                        .contains("<source id=\"1\">", "Question: How do I enable HNSW?"));
    }

    @Test
    void sourcesCarryTheScoreOfEveryStageThatReturnedTheChunk() {
        RetrievalResult retrieval = oneChunk();
        Document chunk = retrieval.documents().getFirst();
        PipelineTrace trace = new PipelineTrace(List.of(
                new PipelineTrace.Stage("vector", 3, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.61))),
                new PipelineTrace.Stage("keyword", 2, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.08))),
                new PipelineTrace.Stage("fusion", 0, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.0325)))),
                4);
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(retrieval.documents(), trace));

        StepVerifier.create(service(new StubChatModel("ok [1]")).answer("q"))
                .assertNext(event -> assertThat(event).isInstanceOfSatisfying(ChatEvent.Sources.class,
                        sources -> assertThat(sources.sources().getFirst().scores())
                                .containsExactly(Map.entry("vector", 0.61), Map.entry("keyword", 0.08),
                                        Map.entry("fusion", 0.0325))))
                .thenConsumeWhile(event -> true)
                .verifyComplete();
    }

    @Test
    void emptyRetrievalAnswersWithoutCallingTheModel() {
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(List.of(), new PipelineTrace(List.of())));
        StubChatModel model = new StubChatModel("should never be streamed");

        StepVerifier.create(service(model).answer("What is the capital of France?"))
                .expectNext(new ChatEvent.Sources(List.of()), new ChatEvent.Token(AnswerService.NO_SOURCES_ANSWER))
                .assertNext(event -> assertThat(event).isInstanceOf(ChatEvent.Done.class))
                .verifyComplete();

        assertThat(model.prompts()).isEmpty();
    }

    @Test
    void modelFailureMidStreamEndsWithAGenericErrorEvent() {
        when(pipeline.retrieve(anyString())).thenReturn(oneChunk());
        StubChatModel model = StubChatModel.failingAfter(new IllegalStateException("rate limited"), "Partial ");

        StepVerifier.create(service(model).answer("q"))
                .expectNextMatches(ChatEvent.Sources.class::isInstance)
                .expectNext(new ChatEvent.Token("Partial "), new ChatEvent.Error(AnswerService.ANSWER_FAILED)) // "rate limited" stays in the log
                .verifyComplete();
    }

    @Test
    void bracesInSourcesAndQuestionsReachTheModelVerbatim() {
        // Retrieved code samples are full of {placeholders}; they must never be read as prompt-template variables.
        Document chunk = Document.builder()
                .text("Config › Templates\n\nUse {name} or {{double}} and ${user.home} in templates.")
                .metadata(Map.<String, Object>of(SOURCE_PATH, "templates.adoc", TITLE, "Config", BREADCRUMB, "Templates"))
                .build();
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(List.of(chunk), new PipelineTrace(List.of())));
        StubChatModel model = new StubChatModel("ok [1]");

        StepVerifier.create(service(model).answer("What does {name} expand to?"))
                .expectNextMatches(ChatEvent.Sources.class::isInstance)
                .expectNext(new ChatEvent.Token("ok [1]"))
                .expectNextMatches(ChatEvent.Done.class::isInstance)
                .verifyComplete();

        assertThat(model.prompts()).singleElement().satisfies(prompt -> assertThat(prompt.getContents())
                .contains("Use {name} or {{double}} and ${user.home} in templates.", "Question: What does {name} expand to?"));
    }

    @Test
    void retrievalFailureBecomesAGenericErrorEvent() {
        when(pipeline.retrieve(anyString())).thenThrow(new IllegalStateException("database down"));

        StepVerifier.create(service(new StubChatModel()).answer("q"))
                .expectNext(new ChatEvent.Error(AnswerService.ANSWER_FAILED))
                .verifyComplete();
    }
}
