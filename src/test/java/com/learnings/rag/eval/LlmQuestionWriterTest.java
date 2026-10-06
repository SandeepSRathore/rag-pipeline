package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.io.ClassPathResource;

import com.learnings.rag.StubChatModel;

class LlmQuestionWriterTest {

    private static final CorpusChunk CHUNK = new CorpusChunk("api/prompt.adoc", "Prompts", "Templates", 3, 120,
            "Prompts › Templates\n\nUse {name} placeholders; ${user.home} is not expanded.");

    private static LlmQuestionWriter writer(StubChatModel model) {
        return new LlmQuestionWriter(ChatClient.builder(model), new ClassPathResource("prompts/golden-question.st"));
    }

    @Test
    void parsesTheStructuredReply() {
        StubChatModel model = new StubChatModel(
                "{\"usable\": true, \"question\": \"How do I fill a placeholder?\", \"referenceAnswer\": \"Pass a value.\"}");

        assertThat(writer(model).write(CHUNK, null))
                .isEqualTo(new GeneratedQuestion(true, "How do I fill a placeholder?", "Pass a value."));
    }

    @Test
    void promptCarriesTheRulesTheChunkVerbatimAndTheJsonFormat() {
        StubChatModel model = new StubChatModel("{\"usable\": false, \"question\": \"\", \"referenceAnswer\": \"\"}");

        writer(model).write(CHUNK, null);

        assertThat(model.prompts()).singleElement().satisfies(prompt -> {
            assertThat(prompt.getSystemMessage().getText()).contains("Use your own words", "usable to false");
            assertThat(prompt.getUserMessage().getText())
                    .contains("Document: Prompts", "Section: Templates",
                            "Use {name} placeholders; ${user.home} is not expanded.")
                    .contains("JSON");
        });
    }

    @Test
    void aRetryNamesThePhraseToAvoid() {
        assertThat(LlmQuestionWriter.userMessage(CHUNK, "use name placeholders"))
                .endsWith("Your previous question copied this phrase from the excerpt: \"use name placeholders\". "
                        + "Write a different question in your own words.");
    }

    @Test
    void introductionChunksAreLabelled() {
        CorpusChunk intro = new CorpusChunk("api/prompt.adoc", "Prompts", "", 0, 90, "Prompts\n\nPrompts guide models.");

        assertThat(LlmQuestionWriter.userMessage(intro, null)).contains("Section: (introduction)");
    }
}
