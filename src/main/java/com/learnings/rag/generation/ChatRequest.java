package com.learnings.rag.generation;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * @param debug also stream the retrieval trace (a {@code trace} event right after {@code sources}). A wrapper, so that
 *        a request without it means false: Jackson 3 rejects a missing primitive (FAIL_ON_NULL_FOR_PRIMITIVES).
 */
public record ChatRequest(@NotBlank @Size(max = 2000) String question, Boolean debug) {

    public ChatRequest {
        debug = Boolean.TRUE.equals(debug);
    }
}
