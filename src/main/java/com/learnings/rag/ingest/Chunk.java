package com.learnings.rag.ingest;

import java.util.List;

/**
 * @param index position within its document
 * @param headingPath headings above the chunk, outermost first (empty for a document preamble)
 * @param body the chunk text without any header
 * @param tokenCount body size in cl100k tokens (the tokenizer of text-embedding-3-small)
 */
public record Chunk(int index, List<String> headingPath, String body, int tokenCount) {

    public String breadcrumb() {
        return String.join(" › ", headingPath);
    }

    /** What gets embedded and stored: a "Title › Section › Subsection" header, a blank line, then the body. */
    public String contextualText(String title) {
        String header = headingPath.isEmpty() ? title : title + " › " + breadcrumb();
        return header + "\n\n" + body;
    }
}
