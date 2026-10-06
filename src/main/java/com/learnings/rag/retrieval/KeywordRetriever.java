package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.json.JsonMapper;

/**
 * Lexical retrieval over the generated {@code content_tsv} column (GIN-indexed, {@code english} configuration).
 * {@code websearch_to_tsquery} parses the question (stop words, stemming, "quoted phrases", -negation) and never
 * fails on user input, but joins the terms with AND, so a natural-language question rarely matches any chunk.
 * Replacing {@code &} with {@code |} asks for chunks matching any term; {@code ts_rank_cd} then ranks chunks that
 * match more terms, closer together, higher. Dotted identifiers stay single lexemes, so exact property and class
 * names still match precisely.
 */
@Component
public class KeywordRetriever {

    private static final TypeReference<Map<String, Object>> METADATA = new TypeReference<>() {
    };

    private static final String SEARCH = """
            SELECT id::text AS id, content, metadata::text AS metadata, ts_rank_cd(content_tsv, q.query) AS rank
            FROM vector_store,
                 (SELECT replace(websearch_to_tsquery('english', :question)::text, '&', '|')::tsquery AS query) q
            WHERE content_tsv @@ q.query AND metadata->>'%s' = :model
            ORDER BY rank DESC, id
            LIMIT :limit
            """.formatted(ChunkMetadata.EMBEDDING_MODEL);

    private final JdbcClient jdbc;
    private final JsonMapper json;
    private final String embeddingModel;

    public KeywordRetriever(JdbcClient jdbc, JsonMapper json, RagProperties properties) {
        this.jdbc = jdbc;
        this.json = json;
        this.embeddingModel = properties.embeddingModel();
    }

    /** Up to {@code limit} chunks, best first; empty when the question has no searchable words. */
    public List<Document> retrieve(String question, int limit) {
        return jdbc.sql(SEARCH)
                .param("question", question)
                .param("model", embeddingModel)
                .param("limit", limit)
                .query((rs, rowNum) -> Document.builder()
                        .id(rs.getString("id"))
                        .text(rs.getString("content"))
                        .metadata(json.readValue(rs.getString("metadata"), METADATA))
                        .score(rs.getDouble("rank"))
                        .build())
                .list();
    }
}
