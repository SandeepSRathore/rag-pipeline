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
 * {@code websearch_to_tsquery} parses the question (stop words, stemming, "quoted phrases") and never fails on user
 * input, but joins the terms with AND, so a natural-language question rarely matches any chunk. Replacing {@code &}
 * with {@code |} asks for chunks matching any term. Dotted identifiers stay single lexemes, so exact property and
 * class names match precisely.
 * <p>
 * Ranking is {@code ts_rank} with normalization 1 (divided by 1 + log of the chunk's length). The spec first fixed
 * {@code ts_rank_cd}, but under OR that scores every occurrence of any term at full weight, so a chunk repeating a
 * common word ("default") outranked the one chunk holding a rare identifier. The M4 eval measured both (see
 * eval/README.md, spec amendment 27). Websearch {@code -negation} is ignored: OR-ed, a negated term would match every
 * chunk that lacks it at score 0, so rows scoring 0 are dropped, and a question of only negated terms finds nothing.
 */
@Component
public class KeywordRetriever {

    private static final TypeReference<Map<String, Object>> METADATA = new TypeReference<>() {
    };

    private static final String SEARCH = """
            SELECT id::text AS id, content, metadata::text AS metadata, ts_rank(content_tsv, q.query, 1) AS rank
            FROM vector_store,
                 (SELECT replace(websearch_to_tsquery('english', :question)::text, '&', '|')::tsquery AS query) q
            WHERE content_tsv @@ q.query AND ts_rank(content_tsv, q.query, 1) > 0 AND metadata->>'%s' = :model
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

    /** Up to {@code limit} chunks, best first, scored by {@code ts_rank}; empty when the question has no searchable words. */
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
