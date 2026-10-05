package com.learnings.rag.retrieval;

import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/** Dense retrieval: cosine similarity over pgvector's HNSW index. Each Document's score is 1 - cosine distance. */
@Component
public class VectorRetriever implements DocumentRetriever {

    private final VectorStore vectorStore;
    private final RagProperties.Retrieval settings;

    public VectorRetriever(VectorStore vectorStore, RagProperties properties) {
        this.vectorStore = vectorStore;
        this.settings = properties.retrieval();
    }

    @Override
    public List<Document> retrieve(Query query) {
        return vectorStore.similaritySearch(SearchRequest.builder()
                .query(query.text())
                .topK(settings.topK())
                .similarityThreshold(settings.similarityThreshold())
                .build());
    }
}
