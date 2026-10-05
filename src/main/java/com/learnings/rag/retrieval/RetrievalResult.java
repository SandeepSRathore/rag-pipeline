package com.learnings.rag.retrieval;

import java.util.List;

import org.springframework.ai.document.Document;

/** @param documents chunks to hand to the model, best first */
public record RetrievalResult(List<Document> documents, PipelineTrace trace) {
}
