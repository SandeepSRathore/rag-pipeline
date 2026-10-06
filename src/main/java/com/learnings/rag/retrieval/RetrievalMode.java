package com.learnings.rag.retrieval;

/** Which retrievers answer a question. HYBRID runs vector and keyword search and fuses them with RRF. */
public enum RetrievalMode {
    VECTOR, KEYWORD, HYBRID
}
