package com.learnings.rag.ingest;

import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

import org.springframework.jdbc.core.RowMapper;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

@Repository
public class SourceDocumentRepository {

    private static final RowMapper<SourceDocument> ROW_MAPPER = (rs, rowNum) -> new SourceDocument(
            rs.getObject("id", UUID.class),
            rs.getString("source_path"),
            rs.getString("title"),
            rs.getString("fingerprint"),
            rs.getInt("chunk_count"),
            SourceDocument.Origin.valueOf(rs.getString("origin")),
            rs.getObject("ingested_at", OffsetDateTime.class).toInstant());

    private final JdbcClient jdbc;

    public SourceDocumentRepository(JdbcClient jdbc) {
        this.jdbc = jdbc;
    }

    public Optional<SourceDocument> findById(UUID id) {
        return jdbc.sql("SELECT * FROM source_document WHERE id = ?").param(id).query(ROW_MAPPER).optional();
    }

    public Optional<SourceDocument> findBySourcePath(String sourcePath) {
        return jdbc.sql("SELECT * FROM source_document WHERE source_path = ?").param(sourcePath).query(ROW_MAPPER).optional();
    }

    public List<SourceDocument> findAll() {
        return jdbc.sql("SELECT * FROM source_document ORDER BY source_path").query(ROW_MAPPER).list();
    }

    public List<SourceDocument> findByOrigin(SourceDocument.Origin origin) {
        return jdbc.sql("SELECT * FROM source_document WHERE origin = ? ORDER BY source_path")
                .param(origin.name()).query(ROW_MAPPER).list();
    }

    public void save(SourceDocument document) {
        jdbc.sql("""
                INSERT INTO source_document (id, source_path, title, fingerprint, chunk_count, origin, ingested_at)
                VALUES (:id, :sourcePath, :title, :fingerprint, :chunkCount, :origin, :ingestedAt)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    fingerprint = EXCLUDED.fingerprint,
                    chunk_count = EXCLUDED.chunk_count,
                    ingested_at = EXCLUDED.ingested_at""")
                .param("id", document.id())
                .param("sourcePath", document.sourcePath())
                .param("title", document.title())
                .param("fingerprint", document.fingerprint())
                .param("chunkCount", document.chunkCount())
                .param("origin", document.origin().name())
                .param("ingestedAt", OffsetDateTime.ofInstant(document.ingestedAt(), ZoneOffset.UTC))
                .update();
    }

    /** IDs of the vector_store rows that belong to a document. */
    public List<String> chunkIds(UUID sourceId) {
        return jdbc.sql("SELECT id::text FROM vector_store WHERE metadata->>'source_id' = ?")
                .param(sourceId.toString()).query(String.class).list();
    }

    public void deleteById(UUID id) {
        jdbc.sql("DELETE FROM source_document WHERE id = ?").param(id).update();
    }
}
