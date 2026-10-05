package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import tools.jackson.databind.json.JsonMapper;

class GoldenSetFileTest {

    private final GoldenSetFile file = new GoldenSetFile(JsonMapper.builder().build());

    @TempDir
    Path dir;

    private static GoldenItem item(String id, String question) {
        return new GoldenItem(id, question, List.of(new ExpectedSource("api/vectordbs/pgvector.adoc", "Indexes")),
                "Use the HNSW index type.", "PGvector › Indexes\n\nHNSW builds a graph.");
    }

    @Test
    void draftRoundTripsAndRecordsTheFileHash() {
        Path path = dir.resolve("eval/golden-set.draft.json");
        List<GoldenItem> items = List.of(item("q01", "How do I enable HNSW?"), item("q02", "Why is IVFFlat slower?"));

        file.writeDraft(path, items, false);
        GoldenSet set = file.read(path);

        assertThat(set.items()).isEqualTo(items);
        assertThat(set.path()).isEqualTo(path);
        assertThat(set.sha256()).hasSize(64);
    }

    @Test
    void refusesToOverwriteAnExistingDraft() throws IOException {
        Path path = dir.resolve("golden-set.draft.json");
        Files.writeString(path, "[\"review edits\"]");

        assertThatThrownBy(() -> file.writeDraft(path, List.of(item("q01", "How do I enable HNSW?")), false))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("rag.eval.golden.overwrite=true");
        assertThat(Files.readString(path)).isEqualTo("[\"review edits\"]");

        file.writeDraft(path, List.of(item("q01", "How do I enable HNSW?")), true);
        assertThat(Files.readString(path)).contains("q01");
    }

    @Test
    void rejectsMisspelledOrMissingFieldsInsteadOfScoringWithDefaults() throws IOException {
        Path path = dir.resolve("golden-set.json");
        Files.writeString(path, """
                [
                  {"id": "q01", "question": "How do I enable HNSW?",
                   "expectedSources": [{"sourcePath": "a.adoc", "sectionPrefx": "Indexes"}]},
                  {"id": "q01", "question": " ",
                   "expectedSource": [{"sourcePath": "b.adoc", "sectionPrefix": ""}]}
                ]""");

        assertThatThrownBy(() -> file.read(path))
                .isInstanceOf(InvalidGoldenSetException.class)
                .hasMessageContaining("q01: a.adoc has no sectionPrefix")
                .hasMessageContaining("q01: duplicate id")
                .hasMessageContaining("q01: missing question")
                .hasMessageContaining("q01: no expectedSources");
    }

    @Test
    void missingFileExplainsHowToCreateIt() {
        assertThatThrownBy(() -> file.read(dir.resolve("golden-set.json")))
                .isInstanceOf(InvalidGoldenSetException.class)
                .hasMessageContaining("'golden' profile");
    }

    @Test
    void malformedJsonIsAnInvalidGoldenSet() throws IOException {
        Path path = dir.resolve("golden-set.json");
        Files.writeString(path, "{ not json");

        assertThatThrownBy(() -> file.read(path))
                .isInstanceOf(InvalidGoldenSetException.class)
                .hasMessageContaining("not a JSON array");
    }

    @Test
    void anEmptySetIsInvalid() throws IOException {
        Path path = dir.resolve("golden-set.json");
        Files.writeString(path, "[]");

        assertThatThrownBy(() -> file.read(path)).hasMessageContaining("the set is empty");
    }
}
