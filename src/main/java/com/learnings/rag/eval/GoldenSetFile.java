package com.learnings.rag.eval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;

import org.springframework.stereotype.Component;

import tools.jackson.core.JacksonException;
import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

/**
 * Reads and validates the golden set, and writes generated drafts. Jackson ignores unknown fields, so a misspelled
 * field in a hand-edited file arrives as null: validation rejects it rather than scoring with a default.
 */
@Component
public class GoldenSetFile {

    private static final TypeReference<List<GoldenItem>> ITEMS = new TypeReference<>() {
    };

    private final JsonMapper json;

    public GoldenSetFile(JsonMapper json) {
        this.json = json;
    }

    public GoldenSet read(Path path) {
        byte[] bytes;
        try {
            bytes = Files.readAllBytes(path);
        }
        catch (NoSuchFileException e) {
            throw new InvalidGoldenSetException(path + " does not exist. Generate a draft with the 'golden' profile, "
                    + "review it, and save it as " + path + " (see eval/README.md).");
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }

        List<GoldenItem> items;
        try {
            items = json.readValue(bytes, ITEMS);
        }
        catch (JacksonException e) {
            throw new InvalidGoldenSetException(path + " is not a JSON array of golden items: " + e.getOriginalMessage());
        }
        List<String> problems = problems(items);
        if (!problems.isEmpty()) {
            throw new InvalidGoldenSetException(path + " has " + problems.size() + " problem(s):\n- "
                    + String.join("\n- ", problems));
        }
        return new GoldenSet(path, sha256(bytes), List.copyOf(items));
    }

    /** Writes a draft for review, pretty-printed. Refuses to replace an existing file unless {@code overwrite}. */
    public void writeDraft(Path path, List<GoldenItem> items, boolean overwrite) {
        ensureDraftWritable(path, overwrite);
        try {
            Files.createDirectories(path.toAbsolutePath().getParent());
            Files.write(path, json.writer().with(SerializationFeature.INDENT_OUTPUT).writeValueAsBytes(items));
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Fails when {@code path} exists, since it may hold review edits, unless {@code overwrite}. */
    public void ensureDraftWritable(Path path, boolean overwrite) {
        if (Files.exists(path) && !overwrite) {
            throw new IllegalStateException(path + " already exists and may hold review edits. Move it away, or run "
                    + "with --rag.eval.golden.overwrite=true to replace it.");
        }
    }

    static List<String> problems(List<GoldenItem> items) {
        List<String> problems = new ArrayList<>();
        if (items == null || items.isEmpty()) {
            problems.add("the set is empty");
            return problems;
        }
        Set<String> ids = new HashSet<>();
        for (int i = 0; i < items.size(); i++) {
            GoldenItem item = items.get(i);
            if (item == null) {
                problems.add("item #" + (i + 1) + ": null");
                continue;
            }
            String label = isBlank(item.id()) ? "item #" + (i + 1) : item.id();
            if (isBlank(item.id())) {
                problems.add(label + ": missing id");
            }
            else if (!ids.add(item.id())) {
                problems.add(label + ": duplicate id");
            }
            if (isBlank(item.question())) {
                problems.add(label + ": missing question");
            }
            if (item.tags().stream().anyMatch(tag -> tag == null || tag.isBlank())) {
                problems.add(label + ": blank tag");
            }
            if (item.expectedSources() == null || item.expectedSources().isEmpty()) {
                problems.add(label + ": no expectedSources");
                continue;
            }
            Set<ExpectedSource> seen = new HashSet<>();
            for (ExpectedSource source : item.expectedSources()) {
                if (source == null || isBlank(source.sourcePath())) {
                    problems.add(label + ": an expected source has no sourcePath");
                }
                else if (source.sectionPrefix() == null) {
                    problems.add(label + ": " + source.sourcePath()
                            + " has no sectionPrefix (use \"\" for the whole page)");
                }
                else if (!seen.add(source)) {
                    problems.add(label + ": duplicate expected source " + source.sourcePath());
                }
            }
        }
        return problems;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        }
        catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is always available", e);
        }
    }
}
