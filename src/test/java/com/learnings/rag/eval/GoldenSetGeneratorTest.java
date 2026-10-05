package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

class GoldenSetGeneratorTest {

    private static CorpusChunk chunk(String page, int index, int tokens) {
        return new CorpusChunk(page, "Title", "Section " + index, index, tokens,
                "Title › Section " + index + "\n\nBody text number " + index + " of " + page + ".");
    }

    /** {@code pages} pages with {@code perPage} chunks each, all 100 tokens. */
    private static List<CorpusChunk> corpus(int pages, int perPage) {
        return IntStream.range(0, pages).boxed()
                .flatMap(p -> IntStream.range(0, perPage).mapToObj(i -> chunk("page-" + p + ".adoc", i, 100)))
                .toList();
    }

    private static EvalProperties.Golden settings(int size) {
        return new EvalProperties.Golden(Path.of("draft.json"), size, 42L, 80, 3, 5, false);
    }

    /** Answers with a fresh question that shares no wording with the chunk, and records what it was asked. */
    private static final class RecordingWriter implements QuestionWriter {

        final List<CorpusChunk> asked = new ArrayList<>();
        final List<String> rejectedPhrases = new ArrayList<>();

        @Override
        public GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase) {
            asked.add(chunk);
            rejectedPhrases.add(rejectedPhrase);
            return new GeneratedQuestion(true, "Question " + asked.size() + "?", "Answer.");
        }
    }

    @Test
    void spreadsQuestionsAcrossPagesBeforeTakingASecondChunkFromAnyPage() {
        RecordingWriter writer = new RecordingWriter();

        List<GoldenItem> items = new GoldenSetGenerator(writer, settings(6)).generate(corpus(4, 3));

        assertThat(items).hasSize(6);
        assertThat(writer.asked.subList(0, 4)).extracting(CorpusChunk::sourcePath).doesNotHaveDuplicates();
        assertThat(items).extracting(item -> item.expectedSources().getFirst().sourcePath())
                .containsOnly("page-0.adoc", "page-1.adoc", "page-2.adoc", "page-3.adoc");
    }

    @Test
    void theSameSeedSamplesTheSameChunks() {
        RecordingWriter first = new RecordingWriter();
        RecordingWriter second = new RecordingWriter();

        new GoldenSetGenerator(first, settings(5)).generate(corpus(4, 3));
        new GoldenSetGenerator(second, settings(5)).generate(corpus(4, 3));

        assertThat(second.asked).isEqualTo(first.asked);
    }

    @Test
    void chunksBelowTheTokenFloorAreNeverAsked() {
        RecordingWriter writer = new RecordingWriter();

        new GoldenSetGenerator(writer, settings(5)).generate(List.of(chunk("a.adoc", 0, 50), chunk("a.adoc", 1, 120)));

        assertThat(writer.asked).extracting(CorpusChunk::chunkIndex).containsExactly(1);
    }

    @Test
    void anUnusableChunkIsReplacedByAnotherChunkOfTheSamePage() {
        List<CorpusChunk> asked = new ArrayList<>();
        QuestionWriter firstUnusable = (chunk, rejected) -> {
            asked.add(chunk);
            return asked.size() == 1 ? new GeneratedQuestion(false, "", "")
                    : new GeneratedQuestion(true, "What does it do?", "It works.");
        };

        List<GoldenItem> items = new GoldenSetGenerator(firstUnusable, settings(1)).generate(corpus(1, 3));

        assertThat(asked).hasSize(2);
        assertThat(items).singleElement().extracting(GoldenItem::question).isEqualTo("What does it do?");
    }

    @Test
    void aQuestionThatCopiesItsChunkIsRewrittenOnce() {
        List<String> rejected = new ArrayList<>();
        QuestionWriter copiesFirst = (chunk, rejectedPhrase) -> {
            rejected.add(rejectedPhrase);
            return rejectedPhrase == null
                    ? new GeneratedQuestion(true, "Is it true that body text number 0 of a.adoc?", "Yes.")
                    : new GeneratedQuestion(true, "What does this setting control?", "It controls it.");
        };

        List<GoldenItem> items = new GoldenSetGenerator(copiesFirst, settings(1)).generate(List.of(chunk("a.adoc", 0, 100)));

        assertThat(rejected).containsExactly(null, "body text number 0 of a.adoc");
        assertThat(items).singleElement().extracting(GoldenItem::question).isEqualTo("What does this setting control?");
    }

    @Test
    void aQuestionThatKeepsCopyingIsDropped() {
        QuestionWriter alwaysCopies = (chunk, rejectedPhrase) ->
                new GeneratedQuestion(true, "Is it true that body text number 0 of a.adoc?", "Yes.");

        assertThat(new GoldenSetGenerator(alwaysCopies, settings(1)).generate(List.of(chunk("a.adoc", 0, 100)))).isEmpty();
    }

    @Test
    void itemsAreSortedByPageAndPositionWithSequentialIdsAndPointAtTheirChunk() {
        List<CorpusChunk> corpus = List.of(chunk("b.adoc", 1, 100), chunk("b.adoc", 0, 100), chunk("a.adoc", 1, 100),
                chunk("a.adoc", 0, 100));

        List<GoldenItem> items = new GoldenSetGenerator(new RecordingWriter(), settings(4)).generate(corpus);

        assertThat(items).extracting(GoldenItem::id).containsExactly("q01", "q02", "q03", "q04");
        assertThat(items).extracting(item -> item.expectedSources().getFirst())
                .containsExactly(new ExpectedSource("a.adoc", "Section 0"), new ExpectedSource("a.adoc", "Section 1"),
                        new ExpectedSource("b.adoc", "Section 0"), new ExpectedSource("b.adoc", "Section 1"));
        assertThat(items.getFirst().sourceExcerpt()).isEqualTo(chunk("a.adoc", 0, 100).content());
        assertThat(items.getFirst().referenceAnswer()).isEqualTo("Answer.");
    }

    @Test
    void abortsAfterThreeConsecutiveWriterFailures() {
        int[] calls = { 0 };
        QuestionWriter unauthorized = (chunk, rejectedPhrase) -> {
            calls[0]++;
            throw new IllegalStateException("401 Unauthorized");
        };

        assertThatThrownBy(() -> new GoldenSetGenerator(unauthorized, settings(5)).generate(corpus(3, 3)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("3 times in a row")
                .hasRootCauseMessage("401 Unauthorized");
        assertThat(calls[0]).isEqualTo(GoldenSetGenerator.MAX_CONSECUTIVE_FAILURES);
    }

    @Test
    void anOccasionalFailureOnlySkipsThatChunk() {
        int[] calls = { 0 };
        QuestionWriter flaky = (chunk, rejectedPhrase) -> {
            if (++calls[0] == 1) {
                throw new IllegalStateException("429 Too Many Requests");
            }
            return new GeneratedQuestion(true, "What happens on retry " + calls[0] + "?", "It retries.");
        };

        assertThat(new GoldenSetGenerator(flaky, settings(2)).generate(corpus(2, 2))).hasSize(2);
    }

    @Test
    void stopsWhenThePagesRunOut() {
        assertThat(new GoldenSetGenerator(new RecordingWriter(), settings(10)).generate(corpus(2, 1))).hasSize(2);
    }

    @Test
    void blankQuestionsAreTreatedAsUnusable() {
        String[] replies = { "  ", "How do I enable it?" };
        int[] calls = { 0 };
        QuestionWriter blankFirst = (chunk, rejectedPhrase) ->
                new GeneratedQuestion(true, replies[Math.min(calls[0]++, 1)], "Like this.");

        List<GoldenItem> items = new GoldenSetGenerator(blankFirst, settings(1)).generate(corpus(1, 2));

        assertThat(items).extracting(GoldenItem::question).containsExactly("How do I enable it?");
    }
}
