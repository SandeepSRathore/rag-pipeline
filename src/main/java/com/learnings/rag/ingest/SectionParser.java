package com.learnings.rag.ingest;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits AsciiDoc or Markdown into sections keyed by their heading path ("Configuration" › "HNSW").
 * Lines inside code listings, literal blocks and tables are body text, never headings.
 */
final class SectionParser {

    private static final Pattern ADOC_HEADING = Pattern.compile("^(={1,6})\\s+(\\S.*)$");
    private static final Pattern MARKDOWN_HEADING = Pattern.compile("^(#{1,6})\\s+(\\S.*)$");
    private static final Pattern ADOC_DELIMITER = Pattern.compile("^(-{4,}|\\.{4,}|\\|===)$");
    /** AsciiDoc attribute entries (:toc:) and block anchors ([[id]], [#id]) carry no meaning for retrieval. */
    private static final Pattern NOISE = Pattern.compile("^(:[!\\w-]+:.*|\\[\\[[^\\]]*\\]\\]|\\[#[\\w-]+\\])\\s*$");
    private static final Pattern LEADING_BLANK_LINES = Pattern.compile("\\A(?:[ \\t]*\\n)+");

    record Section(List<String> path, String body) {
    }

    /** @param title the first level-1 heading, or {@code null} when there is none */
    record ParsedText(String title, List<Section> sections) {
    }

    private SectionParser() {
    }

    static ParsedText parse(String text) {
        String title = null;
        List<Section> sections = new ArrayList<>();
        List<String> path = new ArrayList<>();
        StringBuilder body = new StringBuilder();
        String openFence = null;

        for (String line : text.replace("\r\n", "\n").split("\n", -1)) {
            String fence = fenceKey(line);
            if (openFence != null) {
                body.append(line).append('\n');
                if (openFence.equals(fence)) {
                    openFence = null;
                }
                continue;
            }
            if (fence != null) {
                openFence = fence;
                body.append(line).append('\n');
                continue;
            }
            if (NOISE.matcher(line).matches()) {
                continue;
            }
            Matcher heading = matchHeading(line);
            if (heading == null) {
                body.append(line).append('\n');
                continue;
            }
            int level = heading.group(1).length();
            String name = heading.group(2).strip();
            if (level == 1 && title == null) {
                title = name;
                continue;
            }
            flush(sections, path, body);
            int depth = Math.max(level - 1, 1);
            while (path.size() >= depth) {
                path.removeLast();
            }
            path.add(name);
        }
        flush(sections, path, body);
        return new ParsedText(title, List.copyOf(sections));
    }

    /** The delimiter that opens and closes a listing, literal block, table or Markdown fence; otherwise null. */
    static String fenceKey(String line) {
        String trimmed = line.strip();
        if (trimmed.startsWith("```")) {
            return "```";
        }
        if (trimmed.startsWith("~~~")) {
            return "~~~";
        }
        return ADOC_DELIMITER.matcher(trimmed).matches() ? trimmed : null;
    }

    private static Matcher matchHeading(String line) {
        Matcher adoc = ADOC_HEADING.matcher(line);
        if (adoc.matches()) {
            return adoc;
        }
        Matcher markdown = MARKDOWN_HEADING.matcher(line);
        return markdown.matches() ? markdown : null;
    }

    private static void flush(List<Section> sections, List<String> path, StringBuilder body) {
        String text = LEADING_BLANK_LINES.matcher(body).replaceFirst("").stripTrailing();
        body.setLength(0);
        if (!text.isEmpty()) {
            sections.add(new Section(List.copyOf(path), text));
        }
    }
}
