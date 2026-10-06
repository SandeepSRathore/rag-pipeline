package com.learnings.rag.eval;

/** What the index holds for the current embedding model; recorded in every eval report. */
public record IndexStats(int corpusDocuments, int uploadedDocuments, int chunks) {
}
