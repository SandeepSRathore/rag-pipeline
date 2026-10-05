package com.learnings.rag.generation;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

/**
 * Builds the grounded prompt: fixed rules in the system message; numbered sources plus the question in the
 * user message. Source numbers match {@link SourceRef#n()}, so the UI can resolve the model's [n] citations.
 */
@Component
public class PromptAssembler {

    private final String systemPrompt;

    public PromptAssembler(@Value("classpath:prompts/answer-system.st") Resource systemPrompt) {
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public AssembledPrompt assemble(String question, List<Document> sources) {
        StringBuilder user = new StringBuilder("<sources>\n");
        for (int i = 0; i < sources.size(); i++) {
            user.append("<source id=\"").append(i + 1).append("\">\n")
                    .append(escape(sources.get(i).getText()))
                    .append("\n</source>\n");
        }
        user.append("</sources>\n\nQuestion: ").append(question);
        return new AssembledPrompt(systemPrompt, user.toString());
    }

    /** A retrieved chunk must not be able to close its own tag and pose as instructions. */
    private static String escape(String text) {
        return text.replace("</source", "&lt;/source").replace("<source", "&lt;source");
    }
}
