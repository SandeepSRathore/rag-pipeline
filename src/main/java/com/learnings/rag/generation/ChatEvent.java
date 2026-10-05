package com.learnings.rag.generation;

import java.util.List;
import java.util.stream.IntStream;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.document.Document;

/** The SSE protocol of /api/chat: one Sources, any number of Tokens, then Done, or Error at any point. */
public sealed interface ChatEvent {

    record Sources(List<SourceRef> sources) implements ChatEvent {

        static Sources from(List<Document> documents) {
            return new Sources(IntStream.range(0, documents.size())
                    .mapToObj(i -> SourceRef.of(i + 1, documents.get(i)))
                    .toList());
        }
    }

    record Token(String text) implements ChatEvent {
    }

    /** Token counts are null when no model call was made. */
    record Done(Integer promptTokens, Integer completionTokens, long retrievalMillis, long generationMillis)
            implements ChatEvent {

        static Done of(Usage usage, long retrievalMillis, long generationMillis) {
            return usage == null ? new Done(null, null, retrievalMillis, generationMillis)
                    : new Done(usage.getPromptTokens(), usage.getCompletionTokens(), retrievalMillis, generationMillis);
        }
    }

    record Error(String message) implements ChatEvent {
    }
}
