package com.learnings.rag.ingest;

/** The corpus directory is missing or empty; syncing it would delete every corpus document. */
public class CorpusUnavailableException extends RuntimeException {

    public CorpusUnavailableException(String message) {
        super(message);
    }
}
