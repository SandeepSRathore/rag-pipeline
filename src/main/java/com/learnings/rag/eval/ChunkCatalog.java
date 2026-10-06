package com.learnings.rag.eval;

import static com.learnings.rag.ingest.ChunkMetadata.BREADCRUMB;
import static com.learnings.rag.ingest.ChunkMetadata.CHUNK_INDEX;
import static com.learnings.rag.ingest.ChunkMetadata.EMBEDDING_MODEL;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_ID;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_PATH;
import static com.learnings.rag.ingest.ChunkMetadata.TITLE;
import static com.learnings.rag.ingest.ChunkMetadata.TOKEN_COUNT;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

import com.learnings.rag.config.RagProperties;

/** Read-only views of the index for the eval harness. Only chunks of the current embedding model count. */
@Repository
public class ChunkCatalog {

    private static final String CORPUS_CHUNKS = """
            SELECT v.metadata->>'%s' AS source_path,
                   v.metadata->>'%s' AS title,
                   v.metadata->>'%s' AS breadcrumb,
                   (v.metadata->>'%s')::int AS chunk_index,
                   (v.metadata->>'%s')::int AS token_count,
                   v.content
            FROM vector_store v
            JOIN source_document d ON d.id::text = v.metadata->>'%s'
            WHERE d.origin = 'CORPUS' AND v.metadata->>'%s' = :model
            ORDER BY source_path, chunk_index
            """.formatted(SOURCE_PATH, TITLE, BREADCRUMB, CHUNK_INDEX, TOKEN_COUNT, SOURCE_ID, EMBEDDING_MODEL);

    private static final String STATS = """
            SELECT count(DISTINCT d.id) FILTER (WHERE d.origin = 'CORPUS') AS corpus_documents,
                   count(DISTINCT d.id) FILTER (WHERE d.origin = 'UPLOAD') AS uploaded_documents,
                   count(*) AS chunks
            FROM vector_store v
            JOIN source_document d ON d.id::text = v.metadata->>'%s'
            WHERE v.metadata->>'%s' = :model
            """.formatted(SOURCE_ID, EMBEDDING_MODEL);

    private static final String BREADCRUMBS = """
            SELECT DISTINCT metadata->>'%s' AS source_path, metadata->>'%s' AS breadcrumb
            FROM vector_store
            WHERE metadata->>'%s' = :model
            """.formatted(SOURCE_PATH, BREADCRUMB, EMBEDDING_MODEL);

    private final JdbcClient jdbc;
    private final String embeddingModel;

    public ChunkCatalog(JdbcClient jdbc, RagProperties properties) {
        this.jdbc = jdbc;
        this.embeddingModel = properties.embeddingModel();
    }

    /** Corpus chunks (uploads excluded), ordered by page and then by position in the page. */
    public List<CorpusChunk> corpusChunks() {
        return jdbc.sql(CORPUS_CHUNKS)
                .param("model", embeddingModel)
                .query((rs, rowNum) -> new CorpusChunk(
                        rs.getString("source_path"),
                        rs.getString("title"),
                        rs.getString("breadcrumb"),
                        rs.getInt("chunk_index"),
                        rs.getInt("token_count"),
                        rs.getString("content")))
                .list();
    }

    /** For every page with chunks of the current model (corpus and uploads), the breadcrumbs of those chunks. */
    public Map<String, Set<String>> breadcrumbsByPage() {
        Map<String, Set<String>> pages = new HashMap<>();
        jdbc.sql(BREADCRUMBS)
                .param("model", embeddingModel)
                .query(rs -> {
                    pages.computeIfAbsent(rs.getString("source_path"), path -> new HashSet<>())
                            .add(rs.getString("breadcrumb"));
                });
        return pages;
    }

    public IndexStats stats() {
        return jdbc.sql(STATS)
                .param("model", embeddingModel)
                .query((rs, rowNum) -> new IndexStats(
                        rs.getInt("corpus_documents"),
                        rs.getInt("uploaded_documents"),
                        rs.getInt("chunks")))
                .single();
    }
}
