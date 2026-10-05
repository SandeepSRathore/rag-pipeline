package com.learnings.rag.retrieval;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.stereotype.Service;

/**
 * M2 baseline: a single vector search. Later milestones add query rewriting, multi-query expansion,
 * keyword search, rank fusion and reranking here, each as a traced stage.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;

    public RetrievalPipeline(VectorRetriever vectorRetriever) {
        this.vectorRetriever = vectorRetriever;
    }

    public RetrievalResult retrieve(String question) {
        long start = System.nanoTime();
        List<Document> documents = vectorRetriever.retrieve(new Query(question));
        long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

        PipelineTrace.Stage vector = new PipelineTrace.Stage("vector", elapsed,
                documents.stream().map(PipelineTrace.Hit::of).toList());
        return new RetrievalResult(documents, new PipelineTrace(List.of(vector)));
    }
}
