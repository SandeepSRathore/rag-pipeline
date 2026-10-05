package com.learnings.rag.ingest;

import java.util.List;

public record ChunkedText(String title, List<Chunk> chunks) {
}
