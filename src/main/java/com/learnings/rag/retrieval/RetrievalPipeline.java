package com.learnings.rag.retrieval;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;

/**
 * M2 baseline: a single vector search. Later milestones add query rewriting, multi-query expansion,
 * keyword search, rank fusion and reranking here, each as a traced stage.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.defaults = RetrievalOptions.from(properties);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Document> documents = vectorRetriever.retrieve(new Query(question), options);
        long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

        PipelineTrace.Stage vector = new PipelineTrace.Stage("vector", elapsed,
                documents.stream().map(PipelineTrace.Hit::of).toList());
        return new RetrievalResult(documents, new PipelineTrace(List.of(vector)));
    }
}
