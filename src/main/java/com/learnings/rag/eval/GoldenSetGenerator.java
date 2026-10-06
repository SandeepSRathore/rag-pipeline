package com.learnings.rag.eval;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Drafts a golden set from stored corpus chunks. Chunks are drawn evenly across pages (one per page per round, pages
 * in seeded random order), so a 211-chunk page gets no more questions than a 4-chunk one. Each question must be
 * usable and must not copy the chunk's wording.
 */
@Service
public class GoldenSetGenerator {

    /** This many writer failures in a row means the model is unreachable (bad key, outage), not one bad reply. */
    static final int MAX_CONSECUTIVE_FAILURES = 3;

    private static final Logger log = LoggerFactory.getLogger(GoldenSetGenerator.class);

    private final QuestionWriter writer;
    private final EvalProperties.Golden settings;

    @Autowired
    public GoldenSetGenerator(QuestionWriter writer, EvalProperties properties) {
        this(writer, properties.golden());
    }

    GoldenSetGenerator(QuestionWriter writer, EvalProperties.Golden settings) {
        this.writer = writer;
        this.settings = settings;
    }

    /** Up to {@code size} questions, sorted by page and position, with ids q01, q02, … */
    public List<GoldenItem> generate(List<CorpusChunk> chunks) {
        return new Run(chunks).execute();
    }

    private record Accepted(CorpusChunk chunk, GeneratedQuestion question) {
    }

    /** One generation run: the seeded sampling state and the failure counter. */
    private final class Run {

        private final Map<String, Deque<CorpusChunk>> candidates = new TreeMap<>();
        private final List<String> pages;
        private final List<Accepted> accepted = new ArrayList<>();
        private int consecutiveFailures;

        Run(List<CorpusChunk> chunks) {
            Random random = new Random(settings.seed());
            for (CorpusChunk chunk : chunks) {
                if (chunk.tokenCount() >= settings.minTokens()) {
                    candidates.computeIfAbsent(chunk.sourcePath(), path -> new ArrayDeque<>()).add(chunk);
                }
            }
            candidates.replaceAll((path, pageChunks) -> {
                List<CorpusChunk> shuffled = new ArrayList<>(pageChunks);
                Collections.shuffle(shuffled, random);
                return new ArrayDeque<>(shuffled);
            });
            pages = new ArrayList<>(candidates.keySet());
            Collections.shuffle(pages, random);
        }

        List<GoldenItem> execute() {
            while (accepted.size() < settings.size() && candidates.values().stream().anyMatch(page -> !page.isEmpty())) {
                for (String page : pages) {
                    if (accepted.size() >= settings.size()) {
                        break;
                    }
                    takeOneFrom(candidates.get(page));
                }
            }
            if (accepted.size() < settings.size()) {
                log.warn("Only {} of {} questions could be generated: the pages ran out of usable chunks",
                        accepted.size(), settings.size());
            }
            return toItems();
        }

        private void takeOneFrom(Deque<CorpusChunk> page) {
            for (int attempt = 0; attempt < settings.attemptsPerPage() && !page.isEmpty(); attempt++) {
                CorpusChunk chunk = page.poll();
                GeneratedQuestion question = questionFor(chunk);
                if (question != null) {
                    accepted.add(new Accepted(chunk, question));
                    log.info("{}/{} {} › {}: {}", accepted.size(), settings.size(), chunk.sourcePath(),
                            chunk.breadcrumb(), question.question());
                    return;
                }
            }
        }

        /** A usable question in the writer's own words, or null. */
        private GeneratedQuestion questionFor(CorpusChunk chunk) {
            GeneratedQuestion first = ask(chunk, null);
            if (first == null) {
                return null;
            }
            String copied = copiedPhrase(first, chunk);
            if (copied.isEmpty()) {
                return first;
            }
            GeneratedQuestion second = ask(chunk, copied);
            if (second == null || !copiedPhrase(second, chunk).isEmpty()) {
                log.info("Dropped {} › {}: the question kept copying the chunk's wording", chunk.sourcePath(),
                        chunk.breadcrumb());
                return null;
            }
            return second;
        }

        private String copiedPhrase(GeneratedQuestion question, CorpusChunk chunk) {
            String shared = LexicalOverlap.longestSharedRun(question.question(), chunk.content());
            return LexicalOverlap.wordCount(shared) >= settings.maxSharedWords() ? shared : "";
        }

        private GeneratedQuestion ask(CorpusChunk chunk, String rejectedPhrase) {
            GeneratedQuestion question;
            try {
                question = writer.write(chunk, rejectedPhrase);
                consecutiveFailures = 0;
            }
            catch (RuntimeException e) {
                if (++consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
                    throw new IllegalStateException("The question writer failed " + MAX_CONSECUTIVE_FAILURES
                            + " times in a row; giving up (check OPENAI_API_KEY and the chat model)", e);
                }
                log.warn("Question writer failed for {} › {}; trying another chunk", chunk.sourcePath(),
                        chunk.breadcrumb(), e);
                return null;
            }
            boolean usable = question != null && question.usable() && question.question() != null
                    && !question.question().isBlank();
            return usable ? question : null;
        }

        private List<GoldenItem> toItems() {
            List<Accepted> sorted = accepted.stream()
                    .sorted(Comparator.comparing((Accepted a) -> a.chunk().sourcePath())
                            .thenComparingInt(a -> a.chunk().chunkIndex()))
                    .toList();
            List<GoldenItem> items = new ArrayList<>(sorted.size());
            for (int i = 0; i < sorted.size(); i++) {
                CorpusChunk chunk = sorted.get(i).chunk();
                GeneratedQuestion question = sorted.get(i).question();
                items.add(new GoldenItem("q%02d".formatted(i + 1), question.question().strip(),
                        List.of(new ExpectedSource(chunk.sourcePath(), chunk.breadcrumb())),
                        question.referenceAnswer(), chunk.content()));
            }
            return items;
        }
    }
}
