package com.learnings.rag.eval;

/**
 * What the question writer returns for one chunk.
 *
 * @param usable false when the chunk is not worth a question (mostly code, links, a version table, boilerplate)
 */
public record GeneratedQuestion(boolean usable, String question, String referenceAnswer) {
}
