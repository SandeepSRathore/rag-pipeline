package com.learnings.rag.config;

import java.nio.file.Path;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

import com.learnings.rag.retrieval.RetrievalMode;

/**
 * @param corpusDir directory scanned by {@code POST /api/ingest/corpus}; filled by scripts/fetch-corpus.sh
 * @param embeddingModel identifies the embedding model; part of every document fingerprint, so switching models
 *        re-embeds the corpus instead of mixing incompatible vectors
 * @param chunking how documents are cut into retrievable chunks
 * @param retrieval how many chunks are retrieved per question
 */
@ConfigurationProperties("rag")
public record RagProperties(@DefaultValue("corpus") Path corpusDir,
        @DefaultValue("unknown") String embeddingModel,
        @DefaultValue Chunking chunking,
        @DefaultValue Retrieval retrieval) {

    /**
     * @param maxTokens prose is packed up to this size; code listings and tables are never split and may exceed it
     * @param minTokens chunks smaller than this are merged into the following chunk
     * @param overlapTokens trailing paragraphs up to this size are repeated at the start of the next chunk
     */
    public record Chunking(@DefaultValue("500") int maxTokens,
            @DefaultValue("50") int minTokens,
            @DefaultValue("60") int overlapTokens) {
    }

    /**
     * @param topK number of chunks handed to the model
     * @param similarityThreshold minimum cosine similarity for vector search; 0 keeps every positive similarity
     * @param mode which retrievers run: VECTOR, KEYWORD, or HYBRID (both, fused with reciprocal rank fusion)
     * @param candidates in HYBRID mode, how many chunks each retriever contributes before fusion
     */
    public record Retrieval(@DefaultValue("5") int topK,
            @DefaultValue("0.0") double similarityThreshold,
            @DefaultValue("VECTOR") RetrievalMode mode,
            @DefaultValue("20") int candidates) {
    }
}
