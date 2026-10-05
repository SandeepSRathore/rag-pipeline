package com.learnings.rag.ingest;

import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;

import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.stereotype.Component;

/** Plain-text formats are read directly (so headings survive for the chunker); rich formats go through Tika. */
@Component
public class TextExtractor {

    private static final Set<String> PLAIN_TEXT = Set.of("adoc", "asciidoc", "md", "markdown", "txt");
    private static final Set<String> RICH = Set.of("pdf", "html", "htm", "docx");

    public boolean supports(String filename) {
        String extension = extension(filename);
        return PLAIN_TEXT.contains(extension) || RICH.contains(extension);
    }

    public String extract(String filename, byte[] content) {
        String extension = extension(filename);
        if (PLAIN_TEXT.contains(extension)) {
            return new String(content, StandardCharsets.UTF_8);
        }
        if (!RICH.contains(extension)) {
            throw new IllegalArgumentException("Unsupported file type: " + filename);
        }
        ByteArrayResource resource = new ByteArrayResource(content) {
            @Override
            public String getFilename() {
                return filename; // Tika uses the name to pick a parser
            }
        };
        return new TikaDocumentReader(resource).get().stream()
                .map(Document::getText)
                .filter(Objects::nonNull)
                .collect(Collectors.joining("\n\n"));
    }

    private static String extension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot < 0 ? "" : filename.substring(dot + 1).toLowerCase(Locale.ROOT);
    }
}
