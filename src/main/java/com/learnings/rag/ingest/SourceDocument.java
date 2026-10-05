package com.learnings.rag.ingest;

import java.time.Instant;
import java.util.UUID;

/**
 * @param sourcePath corpus-relative path ("api/vectordbs/pgvector.adoc") or "uploads/{file name}"
 * @param fingerprint sha256 of the chunker settings plus the content; unchanged fingerprint = skip
 */
public record SourceDocument(UUID id, String sourcePath, String title, String fingerprint, int chunkCount,
        Origin origin, Instant ingestedAt) {

    public enum Origin {
        CORPUS, UPLOAD
    }
}
