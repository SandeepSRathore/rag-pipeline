package com.learnings.rag.ingest;

import static java.util.stream.Collectors.joining;
import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.springframework.ai.tokenizer.JTokkitTokenCountEstimator;
import org.springframework.ai.tokenizer.TokenCountEstimator;

import com.learnings.rag.config.RagProperties;

class StructureAwareChunkerTest {

    private static final TokenCountEstimator TOKENS = new JTokkitTokenCountEstimator();

    private static StructureAwareChunker chunker(int maxTokens, int minTokens, int overlapTokens) {
        return new StructureAwareChunker(new RagProperties.Chunking(maxTokens, minTokens, overlapTokens));
    }

    @Test
    void usesTheDocumentTitleAndHeadingBreadcrumbs() {
        ChunkedText result = chunker(200, 0, 0).chunk("""
                = PGvector

                == Configuration

                Set the index type.

                === HNSW

                HNSW builds a graph.
                """, "fallback");

        assertThat(result.title()).isEqualTo("PGvector");
        assertThat(result.chunks()).extracting(Chunk::breadcrumb).containsExactly("Configuration", "Configuration › HNSW");
        assertThat(result.chunks()).extracting(Chunk::index).containsExactly(0, 1);
    }

    @Test
    void fallsBackToTheGivenTitle() {
        assertThat(chunker(200, 0, 0).chunk("Just text.", "notes").title()).isEqualTo("notes");
    }

    @Test
    void blankTextHasNoChunks() {
        assertThat(chunker(200, 0, 0).chunk("  \n\n", "empty").chunks()).isEmpty();
    }

    @Test
    void contextualTextPrefixesTitleAndBreadcrumb() {
        assertThat(new Chunk(0, List.of("Configuration", "HNSW"), "HNSW builds a graph.", 5).contextualText("PGvector"))
                .isEqualTo("PGvector › Configuration › HNSW\n\nHNSW builds a graph.");
        assertThat(new Chunk(0, List.of(), "Intro.", 2).contextualText("PGvector"))
                .isEqualTo("PGvector\n\nIntro.");
    }

    @Test
    void packsLongProseWithOneParagraphOfOverlap() {
        List<String> paragraphs = IntStream.rangeClosed(1, 6)
                .mapToObj(i -> "Paragraph " + i + " explains how the vector store keeps embeddings next to the original text.")
                .toList();
        int t = paragraphs.stream().mapToInt(TOKENS::estimate).max().orElseThrow();
        // Two paragraphs fit in a chunk, three do not; exactly one paragraph fits in the overlap budget.
        int maxTokens = 2 * t + 4;

        List<Chunk> chunks = chunker(maxTokens, 0, t + 2)
                .chunk("== Storage\n\n" + String.join("\n\n", paragraphs), "doc")
                .chunks();

        assertThat(chunks).hasSize(5);
        for (int i = 0; i < chunks.size() - 1; i++) {
            String[] current = chunks.get(i).body().split("\n\n");
            assertThat(chunks.get(i + 1).body()).startsWith(current[current.length - 1]);
        }
        assertThat(chunks).allSatisfy(chunk -> assertThat(chunk.tokenCount()).isLessThanOrEqualTo(maxTokens));
        assertThat(chunks.getLast().body()).endsWith(paragraphs.getLast());
    }

    @Test
    void keepsAnOversizedCodeListingWholeUpToTheHardLimitEvenWithBlankLines() {
        String code = IntStream.range(0, 12).mapToObj(i -> "int value" + i + " = compute(" + i + ");").collect(joining("\n"));
        String listing = "[source,java]\n----\n" + code + "\n\n// trailing comment after a blank line\n----";
        assertThat(TOKENS.estimate(listing)).as("over maxTokens (40), under the hard limit (4 x 40)").isBetween(41, 160);

        List<Chunk> chunks = chunker(40, 0, 10)
                .chunk("== Example\n\nIntro sentence.\n\n" + listing + "\n\nAfter the listing.", "doc")
                .chunks();

        assertThat(chunks).extracting(Chunk::body).containsExactly("Intro sentence.", listing, "After the listing.");
    }

    @Test
    void listingsBeyondTheHardLimitAreSplitAtLineBoundaries() {
        List<String> lines = IntStream.range(0, 60).mapToObj(i -> "int value" + i + " = compute(" + i + ");").toList();
        String listing = "----\n" + String.join("\n", lines) + "\n----";

        List<Chunk> chunks = chunker(40, 0, 0).chunk("== Example\n\n" + listing, "doc").chunks();

        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(chunks).allSatisfy(chunk -> assertThat(chunk.tokenCount()).isLessThanOrEqualTo(40));
        assertThat(chunks.stream().map(Chunk::body).collect(joining("\n")).lines().toList()).containsSubsequence(lines);
    }

    @Test
    void plainTextIgnoresHeadingAndFenceSyntax() {
        String paragraphs = IntStream.range(0, 12)
                .mapToObj(i -> "Paragraph " + i + " describes how the store keeps embeddings next to the text.")
                .collect(joining("\n\n"));

        ChunkedText result = chunker(40, 0, 0).chunkPlain("# Not a heading\n\n" + "-".repeat(30) + "\n\n" + paragraphs, "notes");

        assertThat(result.title()).isEqualTo("notes");
        assertThat(result.chunks()).hasSizeGreaterThan(1);
        assertThat(result.chunks()).allSatisfy(chunk -> {
            assertThat(chunk.tokenCount()).isLessThanOrEqualTo(40);
            assertThat(chunk.breadcrumb()).isEmpty();
        });
        assertThat(result.chunks().getFirst().body()).startsWith("# Not a heading");
    }

    @Test
    void tablesStayWholeEvenWithBlankLinesBetweenRows() {
        String table = "|===\n|Property |Default\n\n|index-type |HNSW\n\n|distance-type |COSINE_DISTANCE\n|===";

        List<Chunk> chunks = chunker(200, 0, 0).chunk("== Properties\n\n" + table, "doc").chunks();

        assertThat(chunks).singleElement().extracting(Chunk::body).isEqualTo(table);
    }

    @Test
    void oversizedProseParagraphFallsBackToTokenSplitting() {
        String paragraph = IntStream.range(0, 120).mapToObj(i -> "word" + i).collect(joining(" "));

        List<Chunk> chunks = chunker(40, 0, 0).chunk("== Long\n\n" + paragraph, "doc").chunks();

        assertThat(chunks).hasSizeGreaterThan(1);
        assertThat(chunks).allSatisfy(chunk -> assertThat(chunk.tokenCount()).isLessThanOrEqualTo(40));
        assertThat(chunks.getFirst().body()).startsWith("word0");
        assertThat(chunks.getLast().body()).endsWith("119");
    }

    @Test
    void smallSectionsMergeIntoTheNextUnderTheirCommonParent() {
        String hnsw = "The HNSW index builds a multilayer graph that gives better recall than IVFFlat at the cost of memory.";

        List<Chunk> chunks = chunker(200, 10, 0).chunk("""
                == Indexes

                === Tiny

                Short.

                === HNSW

                %s
                """.formatted(hnsw), "doc").chunks();

        assertThat(chunks).singleElement().satisfies(chunk -> {
            assertThat(chunk.breadcrumb()).isEqualTo("Indexes");
            assertThat(chunk.body()).isEqualTo("Short.\n\n" + hnsw);
        });
    }

    @Test
    void fingerprintChangesWhenSettingsChange() {
        assertThat(chunker(500, 50, 60).fingerprint())
                .isEqualTo(chunker(500, 50, 60).fingerprint())
                .isNotEqualTo(chunker(400, 50, 60).fingerprint());
    }
}
