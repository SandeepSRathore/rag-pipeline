package com.learnings.rag.ingest;

import static com.learnings.rag.ingest.ChunkMetadata.BREADCRUMB;
import static com.learnings.rag.ingest.ChunkMetadata.CHUNK_INDEX;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_ID;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_PATH;
import static com.learnings.rag.ingest.ChunkMetadata.TITLE;
import static com.learnings.rag.ingest.ChunkMetadata.TOKEN_COUNT;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.transaction.PlatformTransactionManager;
import org.springframework.transaction.support.TransactionTemplate;

import com.learnings.rag.config.RagProperties;

/**
 * Writes one document's chunks to the vector store.
 * <ul>
 * <li>Unchanged content (same fingerprint) is skipped.</li>
 * <li>New or changed content is embedded and inserted <em>before</em> the previous chunks are removed, so the slow
 * OpenAI call never holds a database transaction open, and a failed embedding leaves the previous version
 * searchable. Only the swap (delete previous chunks, update the document row) is transactional; between insert
 * and swap a search may briefly see both versions. If the swap fails, the new chunks are removed again.</li>
 * <li>Ingests and deletes of the same source path run one at a time, so a double-clicked upload or two overlapping
 * corpus syncs cannot both insert chunks for one document (the lock is per JVM; this app runs as one instance).</li>
 * </ul>
 */
@Service
public class DocumentIngestionService {

    private final VectorStore vectorStore;
    private final SourceDocumentRepository documents;
    private final StructureAwareChunker chunker;
    private final String embeddingModel;
    private final TransactionTemplate transaction;
    private final ConcurrentMap<String, Lock> pathLocks = new ConcurrentHashMap<>();

    public DocumentIngestionService(VectorStore vectorStore, SourceDocumentRepository documents,
            StructureAwareChunker chunker, RagProperties properties, PlatformTransactionManager transactionManager) {
        this.vectorStore = vectorStore;
        this.documents = documents;
        this.chunker = chunker;
        this.embeddingModel = properties.embeddingModel();
        this.transaction = new TransactionTemplate(transactionManager);
    }

    public record IngestOutcome(SourceDocument document, Status status) {

        public enum Status {
            ADDED, UPDATED, SKIPPED
        }

        public int chunksWritten() {
            return status == Status.SKIPPED ? 0 : document.chunkCount();
        }
    }

    public IngestOutcome ingest(String sourcePath, String fallbackTitle, String text, SourceDocument.Origin origin) {
        Lock lock = lockFor(sourcePath);
        lock.lock();
        try {
            return ingestLocked(sourcePath, fallbackTitle, text, origin);
        }
        finally {
            lock.unlock();
        }
    }

    private IngestOutcome ingestLocked(String sourcePath, String fallbackTitle, String text,
            SourceDocument.Origin origin) {
        // Vectors from different embedding models live in different spaces and must never be compared,
        // so the model is part of the fingerprint alongside the chunker settings.
        String fingerprint = sha256(embeddingModel + "\n" + chunker.fingerprint() + "\n" + text);
        Optional<SourceDocument> existing = documents.findBySourcePath(sourcePath);
        if (existing.isPresent() && existing.get().fingerprint().equals(fingerprint)) {
            return new IngestOutcome(existing.get(), IngestOutcome.Status.SKIPPED);
        }

        UUID id = existing.map(SourceDocument::id).orElseGet(UUID::randomUUID);
        List<String> previousChunkIds = documents.chunkIds(id);
        ChunkedText chunked = chunker.chunk(text, fallbackTitle);
        List<Document> chunks = chunked.chunks().stream()
                .map(chunk -> toDocument(id, sourcePath, chunked.title(), chunk))
                .toList();
        insertOrUndo(chunks);

        SourceDocument saved = new SourceDocument(id, sourcePath, chunked.title(), fingerprint, chunks.size(), origin,
                Instant.now());
        try {
            transaction.executeWithoutResult(status -> {
                deleteChunks(previousChunkIds);
                documents.save(saved);
            });
        }
        catch (RuntimeException e) {
            // The swap rolled back, so the previous version is intact; the chunks just inserted would be orphans.
            deleteChunksAfterFailure(chunks, e);
            throw e;
        }
        return new IngestOutcome(saved, existing.isPresent() ? IngestOutcome.Status.UPDATED : IngestOutcome.Status.ADDED);
    }

    public boolean delete(UUID id) {
        Optional<SourceDocument> document = documents.findById(id);
        if (document.isEmpty()) {
            return false;
        }
        Lock lock = lockFor(document.get().sourcePath());
        lock.lock();
        try {
            return Boolean.TRUE.equals(transaction.execute(status -> {
                if (documents.findById(id).isEmpty()) {
                    return false; // deleted while we waited for the lock
                }
                deleteChunks(documents.chunkIds(id));
                documents.deleteById(id);
                return true;
            }));
        }
        finally {
            lock.unlock();
        }
    }

    private Lock lockFor(String sourcePath) {
        return pathLocks.computeIfAbsent(sourcePath, path -> new ReentrantLock());
    }

    /** Embeds and inserts the chunks; if that fails part-way, removes whatever was inserted and rethrows. */
    private void insertOrUndo(List<Document> chunks) {
        if (chunks.isEmpty()) {
            return;
        }
        try {
            vectorStore.add(chunks); // embeds every chunk: the slow, failure-prone OpenAI call
        }
        catch (RuntimeException e) {
            deleteChunksAfterFailure(chunks, e);
            throw e;
        }
    }

    private void deleteChunksAfterFailure(List<Document> chunks, RuntimeException failure) {
        try {
            deleteChunks(chunks.stream().map(Document::getId).toList());
        }
        catch (RuntimeException cleanupFailure) {
            failure.addSuppressed(cleanupFailure);
        }
    }

    private void deleteChunks(List<String> chunkIds) {
        if (!chunkIds.isEmpty()) {
            vectorStore.delete(chunkIds);
        }
    }

    private static Document toDocument(UUID sourceId, String sourcePath, String title, Chunk chunk) {
        return Document.builder()
                .text(chunk.contextualText(title))
                .metadata(Map.<String, Object>of(
                        SOURCE_ID, sourceId.toString(),
                        SOURCE_PATH, sourcePath,
                        TITLE, title,
                        BREADCRUMB, chunk.breadcrumb(),
                        CHUNK_INDEX, chunk.index(),
                        TOKEN_COUNT, chunk.tokenCount()))
                .build();
    }

    static String sha256(String value) {
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(digest);
        }
        catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is always available", e);
        }
    }
}
