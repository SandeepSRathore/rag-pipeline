package com.learnings.rag.eval;

/**
 * Where the answer to a golden question lives: a page plus a heading-path prefix. Labels point at sections, not
 * chunk ids, so they stay valid when the corpus is re-chunked.
 *
 * @param sourcePath corpus-relative page, e.g. {@code api/vectordbs/pgvector.adoc}
 * @param sectionPrefix heading path the chunk must sit in, e.g. {@code Auto-Configuration › Configuration properties};
 *        "" accepts any chunk of the page
 */
public record ExpectedSource(String sourcePath, String sectionPrefix) {

    private static final String SEPARATOR = " › ";

    /** True when a chunk at this path and breadcrumb lies inside the expected section, or is the section itself. */
    public boolean matches(String chunkSourcePath, String chunkBreadcrumb) {
        if (!sourcePath.equals(chunkSourcePath)) {
            return false;
        }
        return sectionPrefix.isEmpty() || chunkBreadcrumb.equals(sectionPrefix)
                || chunkBreadcrumb.startsWith(sectionPrefix + SEPARATOR);
    }
}
