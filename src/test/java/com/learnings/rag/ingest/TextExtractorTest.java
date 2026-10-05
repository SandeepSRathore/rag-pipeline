package com.learnings.rag.ingest;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

class TextExtractorTest {

    private final TextExtractor extractor = new TextExtractor();

    @Test
    void plainTextFormatsAreReadAsUtf8() {
        assertThat(extractor.extract("notes.md", "# Café\n\nbody".getBytes(StandardCharsets.UTF_8)))
                .isEqualTo("# Café\n\nbody");
    }

    @Test
    void htmlGoesThroughTikaAndLosesItsMarkup() {
        String text = extractor.extract("page.html",
                "<html><body><h1>Hello</h1><p>Vector stores</p></body></html>".getBytes(StandardCharsets.UTF_8));

        assertThat(text).contains("Hello", "Vector stores").doesNotContain("<p>");
    }

    @Test
    void supportIsDecidedByExtensionCaseInsensitively() {
        assertThat(extractor.supports("GUIDE.MD")).isTrue();
        assertThat(extractor.supports("report.pdf")).isTrue();
        assertThat(extractor.supports("diagram.png")).isFalse();
        assertThat(extractor.supports("README")).isFalse();
    }

    @Test
    void unsupportedTypesAreRejected() {
        assertThatThrownBy(() -> extractor.extract("diagram.png", new byte[] { 1 }))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
