package com.learnings.rag.eval;

/** Writes one evaluation question about a chunk. The production implementation is {@link LlmQuestionWriter}. */
public interface QuestionWriter {

    /**
     * @param rejectedPhrase a phrase the previous attempt copied from the chunk and must avoid; null on the first
     *        attempt
     */
    GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase);
}
