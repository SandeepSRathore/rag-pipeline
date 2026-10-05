package com.learnings.rag.eval;

/** The golden set file is missing, unreadable, or has items that cannot be scored. */
public class InvalidGoldenSetException extends RuntimeException {

    public InvalidGoldenSetException(String message) {
        super(message);
    }
}
