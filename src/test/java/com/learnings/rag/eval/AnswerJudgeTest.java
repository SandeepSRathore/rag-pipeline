package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.eval.AnswerJudge.Verdict;

class AnswerJudgeTest {

    private static final List<String> SOURCES = List.of("PGvector › Indexes\n\nHNSW is the default index type.");

    private static AnswerJudge judge(ChatModel model) {
        return new AnswerJudge(ChatClient.builder(model).build());
    }

    @Test
    void yesInAnyCasePassesAndNoFails() {
        assertThat(judge(new StubChatModel("yes")).faithful("HNSW is the default [1].", SOURCES)).isEqualTo(Verdict.PASS);
        assertThat(judge(new StubChatModel("YES")).relevant("Default index?", "HNSW [1].", SOURCES)).isEqualTo(Verdict.PASS);
        assertThat(judge(new StubChatModel("No")).correct("IVFFlat [1].", "HNSW is the default.")).isEqualTo(Verdict.FAIL);
    }

    @Test
    void aDecoratedYesFails() {
        // Spring AI's evaluators accept exactly "yes" in any case; pinned so that an upgrade changing it is noticed.
        assertThat(judge(new StubChatModel("Yes.")).faithful("HNSW is the default [1].", SOURCES)).isEqualTo(Verdict.FAIL);
    }

    @Test
    void aFailingJudgeIsAnErrorNotAFail() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        assertThat(judge(unavailable).faithful("HNSW [1].", SOURCES)).isEqualTo(Verdict.ERROR);
    }

    @Test
    void faithfulnessChecksTheAnswerAgainstItsSources() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).faithful("HNSW is the default [1].", SOURCES);

        assertThat(model.prompts().getFirst().getContents())
                .containsSubsequence("Document:", "HNSW is the default index type.", "Claim:", "HNSW is the default [1].");
    }

    @Test
    void correctnessChecksTheReferenceAnswerAgainstTheAnswer() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).correct("Set index-type to HNSW [1].", "HNSW is the default.");

        assertThat(model.prompts().getFirst().getContents())
                .containsSubsequence("Document:", "Set index-type to HNSW [1].", "Claim:", "HNSW is the default.");
    }

    @Test
    void relevancySeesTheQuestionTheAnswerAndTheSources() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).relevant("Which index is the default?", "HNSW [1].", SOURCES);

        assertThat(model.prompts().getFirst().getContents()).containsSubsequence("Query:", "Which index is the default?",
                "Response:", "HNSW [1].", "Context:", "HNSW is the default index type.");
    }

    @Test
    void bracesAndCodeReachTheJudgesVerbatim() {
        StubChatModel model = new StubChatModel("yes");
        List<String> sources = List.of("Use {name} or {{double}} and ${user.home} in templates.");

        assertThat(judge(model).faithful("Write `{name}` [1].", sources)).isEqualTo(Verdict.PASS);
        assertThat(judge(model).relevant("What does {name} do?", "Write `{name}` [1].", sources)).isEqualTo(Verdict.PASS);

        assertThat(model.prompts()).allSatisfy(prompt -> assertThat(prompt.getContents())
                .contains("Use {name} or {{double}} and ${user.home} in templates.", "Write `{name}` [1]."));
    }
}
