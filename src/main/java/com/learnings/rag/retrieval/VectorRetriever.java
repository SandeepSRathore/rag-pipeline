package com.learnings.rag.retrieval;

import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;

/**
 * Dense retrieval: cosine similarity over pgvector's HNSW index. Each Document's score is 1 - cosine distance.
 * Only chunks embedded by the configured model are searched: vectors from another model (an upload from before a
 * model switch, or the not-yet-re-ingested part of the corpus) are not comparable with the query vector.
 */
@Component
public class VectorRetriever implements DocumentRetriever {

    private final VectorStore vectorStore;
    private final RagProperties.Retrieval settings;
    private final Filter.Expression currentModel;

    public VectorRetriever(VectorStore vectorStore, RagProperties properties) {
        this.vectorStore = vectorStore;
        this.settings = properties.retrieval();
        this.currentModel = new FilterExpressionBuilder()
                .eq(ChunkMetadata.EMBEDDING_MODEL, properties.embeddingModel())
                .build();
    }

    @Override
    public List<Document> retrieve(Query query) {
        return vectorStore.similaritySearch(SearchRequest.builder()
                .query(query.text())
                .topK(settings.topK())
                .similarityThreshold(settings.similarityThreshold())
                .filterExpression(currentModel)
                .build());
    }
}
