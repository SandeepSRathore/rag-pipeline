package com.learnings.rag.eval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

/** Asks the chat model for one question per chunk, as structured output ({@link GeneratedQuestion}). */
@Component
public class LlmQuestionWriter implements QuestionWriter {

    private final ChatClient chatClient;
    private final String systemPrompt;

    public LlmQuestionWriter(ChatClient.Builder chatClientBuilder,
            @Value("classpath:prompts/golden-question.st") Resource systemPrompt) {
        this.chatClient = chatClientBuilder.build();
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase) {
        // Message objects, not template strings: chunks are full of {braces} from code samples.
        return chatClient
                .prompt(new Prompt(List.of(new SystemMessage(systemPrompt),
                        new UserMessage(userMessage(chunk, rejectedPhrase)))))
                .call()
                .entity(GeneratedQuestion.class);
    }

    static String userMessage(CorpusChunk chunk, String rejectedPhrase) {
        StringBuilder text = new StringBuilder()
                .append("Document: ").append(chunk.title()).append('\n')
                .append("Section: ").append(chunk.breadcrumb().isEmpty() ? "(introduction)" : chunk.breadcrumb())
                .append("\n\n<excerpt>\n").append(chunk.content()).append("\n</excerpt>");
        if (rejectedPhrase != null) {
            text.append("\n\nYour previous question copied this phrase from the excerpt: \"").append(rejectedPhrase)
                    .append("\". Write a different question in your own words.");
        }
        return text.toString();
    }
}
