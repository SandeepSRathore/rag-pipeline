package com.learnings.rag.generation;

import java.util.List;
import java.util.stream.IntStream;

import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.document.Document;

import com.learnings.rag.retrieval.PipelineTrace;
import com.learnings.rag.retrieval.RetrievalResult;

/**
 * The SSE protocol of /api/chat: one Sources, a Trace when debugging, any number of Tokens, then Done, or Error at any
 * point.
 */
public sealed interface ChatEvent {

    record Sources(List<SourceRef> sources) implements ChatEvent {

        static Sources from(RetrievalResult retrieval) {
            List<Document> documents = retrieval.documents();
            return new Sources(IntStream.range(0, documents.size())
                    .mapToObj(i -> SourceRef.of(i + 1, documents.get(i), retrieval.trace()))
                    .toList());
        }
    }

    /** Every retrieval stage with its hits and timing; sent only when the request asks for debugging. */
    record Trace(PipelineTrace trace) implements ChatEvent {
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
