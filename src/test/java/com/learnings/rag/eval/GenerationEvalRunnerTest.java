package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.AnswerJudge.Verdict;
import com.learnings.rag.eval.GenerationReport.Group;
import com.learnings.rag.eval.GenerationReport.Item;
import com.learnings.rag.eval.GenerationReport.Outcome;
import com.learnings.rag.eval.GenerationReport.Rate;
import com.learnings.rag.generation.AnswerService;
import com.learnings.rag.generation.ChatEvent;
import com.learnings.rag.generation.SourceRef;
import com.learnings.rag.retrieval.RetrievalMode;

import reactor.core.publisher.Flux;

class GenerationEvalRunnerTest {

    private final AnswerService answers = mock(AnswerService.class);
    private final StubChatModel judgeModel = new StubChatModel("yes");

    private GenerationEvalRunner runner(ChatModel judge) {
        RagProperties properties = new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, "gpt-4.1-mini",
                        new RagProperties.Rerank(true, 6)));
        return new GenerationEvalRunner(answers, new AnswerJudge(ChatClient.builder(judge).build()), properties,
                "gpt-5-mini");
    }

    private static GoldenItem answerable(String id) {
        return new GoldenItem(id, id + "?", List.of(new ExpectedSource("a.adoc", "")), "HNSW is the default.", null);
    }

    private static GoldenItem unanswerable(String id) {
        return new GoldenItem(id, id + "?", List.of(), "Not covered.", null, List.of(GoldenItem.UNANSWERABLE));
    }

    private static List<SourceRef> sources(int count) {
        return IntStream.rangeClosed(1, count)
                .mapToObj(n -> new SourceRef(n, "a.adoc", "A", "S" + n, 8.0, "source text " + n))
                .toList();
    }

    private static Flux<ChatEvent> answered(String text, int sourceCount) {
        return Flux.just(new ChatEvent.Sources(sources(sourceCount)), new ChatEvent.Token(text),
                new ChatEvent.Done(100, 20, 5, 900));
    }

    @Test
    void answeredQuestionsAreJudgedAndTheirCitationsChecked() {
        when(answers.answer("q1?")).thenReturn(answered("HNSW is the default [1].", 2));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1")));

        Item item = report.items().getFirst();
        assertThat(item.outcome()).isEqualTo(Outcome.ANSWERED);
        assertThat(List.of(item.faithful(), item.relevant(), item.correct())).containsOnly(Verdict.PASS);
        assertThat(item.citations().valid()).isTrue();
        assertThat(judgeModel.prompts()).hasSize(3);
        assertThat(report.answerable()).isEqualTo(new Group(1, 1, 0, 0, 0, new Rate(1, 1, 0), new Rate(1, 1, 0),
                new Rate(1, 1, 0), new Rate(1, 1, 0)));
        assertThat(report.p50GenerationMillis()).isEqualTo(900);
        assertThat(report.answerModel()).isEqualTo("gpt-5-mini");
        assertThat(report.judgeModel()).isEqualTo("gpt-4.1-mini");
    }

    @Test
    void refusalsAreClassifiedNotJudged() {
        when(answers.answer("q1?")).thenReturn(Flux.just(new ChatEvent.Sources(List.of()),
                new ChatEvent.Token(AnswerService.NO_SOURCES_ANSWER), new ChatEvent.Done(null, null, 5, 0)));
        when(answers.answer("q2?")).thenReturn(answered("Direct answer: I couldn't find this in the indexed documentation.", 3));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1"), answerable("q2")));

        assertThat(report.items()).extracting(Item::outcome)
                .containsExactly(Outcome.REFUSED_BY_RETRIEVAL, Outcome.REFUSED_BY_MODEL);
        assertThat(report.items()).allSatisfy(item -> assertThat(item.faithful()).isNull());
        assertThat(judgeModel.prompts()).isEmpty();
        assertThat(report.answerable()).isEqualTo(new Group(2, 0, 1, 1, 0, new Rate(0, 0, 0), new Rate(0, 0, 0),
                new Rate(0, 0, 0), new Rate(0, 0, 0)));
    }

    @Test
    void aFailedOrEmptyAnswerIsAFailureNotARefusal() {
        when(answers.answer("q1?")).thenReturn(Flux.just(new ChatEvent.Sources(sources(2)), new ChatEvent.Token("Partial "),
                new ChatEvent.Error(AnswerService.ANSWER_FAILED)));
        when(answers.answer("q2?")).thenReturn(Flux.just(new ChatEvent.Sources(sources(2)),
                new ChatEvent.Done(100, 0, 5, 300)));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1"), answerable("q2")));

        assertThat(report.items()).extracting(Item::outcome).containsExactly(Outcome.FAILED, Outcome.FAILED);
        assertThat(report.answerable().failed()).isEqualTo(2);
        assertThat(judgeModel.prompts()).isEmpty();
    }

    @Test
    void anAnsweredUnanswerableQuestionIsJudgedButNotForCorrectness() {
        when(answers.answer("u1?")).thenReturn(answered("Use spring.ai.openai.chat.options.model [1].", 2));

        GenerationReport report = runner(judgeModel).run(List.of(unanswerable("u1")));

        assertThat(report.items().getFirst().correct()).isNull();
        assertThat(judgeModel.prompts()).hasSize(2);
        assertThat(report.unanswerable().answered()).isEqualTo(1);
        assertThat(report.unanswerable().correct()).isEqualTo(new Rate(0, 0, 0));
    }

    @Test
    void failedChecksAndJudgeErrorsAreCountedApart() {
        when(answers.answer("q1?")).thenReturn(answered("See [4].", 2));
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        GenerationReport failing = runner(new StubChatModel("no")).run(List.of(answerable("q1")));
        GenerationReport erroring = runner(unavailable).run(List.of(answerable("q1")));

        assertThat(failing.answerable().faithful()).isEqualTo(new Rate(0, 1, 0));
        assertThat(failing.answerable().citationsValid()).isEqualTo(new Rate(0, 1, 0));
        assertThat(erroring.answerable().faithful()).isEqualTo(new Rate(0, 0, 1));
        assertThat(erroring.judgeErrors()).isEqualTo(3);
    }
}
