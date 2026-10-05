# RAG Pipeline M3: Evaluation Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the evaluation harness: generate a golden set of questions from the indexed corpus, pause for a human to review it, then score retrieval against it (hit@5, recall@5, MRR@10, p50/p95 latency). The first report records the vector-only baseline.

**Architecture:** A new `eval` package.
- `GoldenSetGenerator` samples stored corpus chunks evenly across pages and asks an LLM (`LlmQuestionWriter`, structured output) for one developer-style question per chunk. It rejects questions that copy the chunk's wording.
- The draft goes to `eval/golden-set.draft.json`; a human reviews it into `eval/golden-set.json`.
- `EvalRunner` runs every golden question through `RetrievalPipeline` with per-call `RetrievalOptions` (top 10), scores the result with `RetrievalMetrics`, and `ReportWriter` writes Markdown + JSON reports.
- Both jobs run as non-web Spring profiles (`golden`, `eval`) that exit when done.

**Tech Stack:** Java 25, Spring Boot 4.1.1, Spring AI 2.0.1 (ChatClient structured output via `.entity()`), Jackson 3 (`tools.jackson.*`), JdbcClient, Testcontainers pgvector, JUnit 5, AssertJ.

**Spec:** `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md` (section "Evaluation harness", milestone M3, amendments 1–20).

## Global Constraints

- Java 25; Spring Boot parent `4.1.1`; `spring-ai-bom` `2.0.1`; build with the Maven wrapper (`./mvnw`).
- Root package `com.learnings.rag`. Configuration via `@ConfigurationProperties` records under the `rag.*` prefix.
- Jackson 3 only (`tools.jackson.*`); never import `com.fasterxml.jackson.databind`.
- Prompts live in `src/main/resources/prompts/*.st`. Prompts are passed as `SystemMessage`/`UserMessage` objects, never as templates (chunks are full of `{braces}`).
- Chunk metadata keys are only referenced through `ChunkMetadata` constants (main code).
- `*Test` = pure unit tests that need no Docker (`./mvnw test`). `*IT` = Docker/Testcontainers tests (`./mvnw verify`).
- Tests never call OpenAI. `FakeEmbeddingModel` and `StubChatModel` stand in. Only the `golden` and `eval` profile runs call OpenAI.
- Golden-set labels point at `sourcePath` + `sectionPrefix` (heading path), never at chunk ids, so they survive re-chunking.
- Metrics are fixed by the spec: hit@5, recall@5, MRR@10, p50/p95 latency.
- Learning project: no auth, rate limits or async jobs. The app listens on `127.0.0.1:8081`.
- Work on a feature branch, not `master`.

**Spec amendments made while planning (also recorded in the spec):**
15. The golden set is a JSON array of `{id, question, expectedSources: [{sourcePath, sectionPrefix}], referenceAnswer, sourceExcerpt}`.
    - `sectionPrefix` matches at heading boundaries, and `""` accepts any chunk of the page.
    - `sourceExcerpt` is reviewer context only and is not scored.
    - The draft holds 40 questions so that review can cut it to 30 or more.
16. **Leakage guard:** a generated question that shares 5 or more consecutive words with its chunk is rewritten once, then dropped. Identifiers such as `spring.ai.vectorstore.pgvector.index-type` count as one word.
17. **One configuration in M3:** M3 compares a single configuration, `vector`. Per-call `RetrievalOptions` let the eval retrieve the top 10 (for MRR@10) while chat keeps the top 5. M4–M6 add their configurations to `EvalConfig.all`.
18. **Eval preconditions and report contents:**
    - The eval refuses to run on an empty index, or when an expected `sourcePath` is not indexed.
    - Reports record the golden set's sha256, the index composition (with a warning when uploads are present), the embedding model and the chunking settings.
19. **Generator safety:**
    - The generator refuses to overwrite an existing draft unless `rag.eval.golden.overwrite=true`.
    - It aborts after 3 consecutive LLM failures.
20. **Running the jobs:** `golden` and `eval` run as non-web profiles: `./mvnw spring-boot:run -Dspring-boot.run.profiles=golden|eval`.

## Review Focus

Inputs the spec implies but doesn't spell out, most likely to bite first. Each one gets a test in the task that owns it:

1. **Re-running the generator while a draft is under review** must not wipe the reviewer's edits. Covered by `refusesToOverwriteAnExistingDraft` (Task 2); the command also checks this before spending any LLM calls (Task 8).
2. **Typos in the hand-edited `golden-set.json`** must be rejected, listing the item ids, instead of being silently scored with defaults. Typos here means a misspelled field (Jackson 3 ignores unknown fields), a missing `sectionPrefix` or a duplicate id. Covered by `rejectsMisspelledOrMissingFieldsInsteadOfScoringWithDefaults` (Task 2).
3. **OpenAI failing during generation** (invalid key, outage, 429) must stop after a few consecutive failures. It must not churn through every page and leave a near-empty draft. Covered by `abortsAfterThreeConsecutiveWriterFailures` (Task 5).
4. **A golden item pointing at a page that isn't indexed** (renamed page, typo, corpus never ingested) must fail the eval, listing the paths, instead of scoring 0. Covered by `expectedSourcesThatAreNotIndexedFailTheRunInsteadOfScoringZero` and `anEmptyIndexFailsTheRun` (Task 7).
5. **Section matching must respect heading boundaries:** `Index` must not match `Indexes › HNSW`, and a coarser parent chunk must not count for a deeper expectation. Covered by `respectsHeadingBoundaries` and `aParentSectionChunkDoesNotCountForADeeperExpectation` (Task 2).

---

## File map

```
src/main/java/com/learnings/rag/retrieval/
  RetrievalOptions.java              per-call topK + threshold (new)
  VectorRetriever.java, RetrievalPipeline.java   accept RetrievalOptions (modify)
src/main/java/com/learnings/rag/eval/
  EvalProperties.java                rag.eval.* (golden-set path, reports dir, generator settings)
  ExpectedSource.java                sourcePath + sectionPrefix, heading-boundary matching
  GoldenItem.java, GoldenSet.java, InvalidGoldenSetException.java
  GoldenSetFile.java                 read + validate the golden set; write the draft
  RetrievalMetrics.java              hit@5, recall@5, MRR@10, nearest-rank percentiles
  CorpusChunk.java, IndexStats.java, ChunkCatalog.java   stored chunks + index composition (JdbcClient)
  LexicalOverlap.java                longest shared word run (leakage guard)
  GeneratedQuestion.java, QuestionWriter.java, GoldenSetGenerator.java
  LlmQuestionWriter.java             ChatClient structured output
  EvalConfig.java, EvalReport.java, EvalRunner.java, ReportWriter.java
  GoldenSetCommand.java, EvalCommand.java   @Profile("golden") / @Profile("eval") ApplicationRunners
src/main/resources/
  prompts/golden-question.st
  application.yml (rag.eval block), application-golden.yml, application-eval.yml
src/test/java/com/learnings/rag/
  retrieval/RetrievalOptionsTest.java, retrieval/RetrievalPipelineIT.java (modify)
  eval/{ExpectedSourceTest, GoldenSetFileTest, RetrievalMetricsTest, LexicalOverlapTest,
        GoldenSetGeneratorTest, LlmQuestionWriterTest, ReportWriterTest}.java
  eval/{ChunkCatalogIT, EvalRunnerIT}.java
eval/README.md                       how to generate, review and run; baseline
eval/golden-set.json                 the reviewed set (Task 9, after the human review)
README.md                            Evaluation section, status
```

---

### Task 1: Per-call retrieval options (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/retrieval/RetrievalOptions.java`
- Modify: `src/main/java/com/learnings/rag/retrieval/VectorRetriever.java`, `src/main/java/com/learnings/rag/retrieval/RetrievalPipeline.java`
- Test: `src/test/java/com/learnings/rag/retrieval/RetrievalOptionsTest.java`, `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineIT.java`

**Interfaces:**
- Consumes: `RagProperties(Path corpusDir, String embeddingModel, Chunking chunking, Retrieval retrieval)`, `RagProperties.Retrieval(int topK, double similarityThreshold)`.
- Produces:
  - `record RetrievalOptions(int topK, double similarityThreshold)` with `static RetrievalOptions from(RagProperties)` and `RetrievalOptions withTopK(int)`. It rejects `topK < 1`.
  - `VectorRetriever.retrieve(Query, RetrievalOptions)`.
  - `RetrievalPipeline.retrieve(String question, RetrievalOptions options)`. `retrieve(String)` keeps working with the defaults, and `AnswerService` keeps calling it.

- [ ] **Step 1: Write the failing unit test**

`src/test/java/com/learnings/rag/retrieval/RetrievalOptionsTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import com.learnings.rag.config.RagProperties;

class RetrievalOptionsTest {

    private static RagProperties properties(int topK, double threshold) {
        return new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(topK, threshold));
    }

    @Test
    void defaultsComeFromTheRetrievalProperties() {
        assertThat(RetrievalOptions.from(properties(5, 0.2))).isEqualTo(new RetrievalOptions(5, 0.2));
    }

    @Test
    void withTopKKeepsTheThreshold() {
        assertThat(new RetrievalOptions(5, 0.2).withTopK(10)).isEqualTo(new RetrievalOptions(10, 0.2));
    }

    @Test
    void topKMustBePositive() {
        assertThatThrownBy(() -> new RetrievalOptions(0, 0.0))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("topK");
    }
}
```

- [ ] **Step 2: Add the failing integration test**

In `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineIT.java`, add this test after `ranksThePgvectorChunkFirstForAnIndexQuestion`:

```java
    @Test
    void topKCanBeOverriddenPerCall() {
        String question = "Which index type is HNSW and how does it build its graph?";
        int indexed = jdbc.sql("SELECT count(*)::int FROM vector_store").query(Integer.class).single();
        assertThat(indexed).as("the fixture corpus must exceed the default top-5").isGreaterThan(5);

        assertThat(pipeline.retrieve(question, RetrievalOptions.from(properties).withTopK(1)).documents()).hasSize(1);
        assertThat(pipeline.retrieve(question, RetrievalOptions.from(properties).withTopK(10)).documents())
                .hasSize(Math.min(10, indexed));
    }
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `./mvnw -q test -Dtest=RetrievalOptionsTest`
Expected: compilation FAILURE, `cannot find symbol: class RetrievalOptions`.

- [ ] **Step 4: Implement `RetrievalOptions` and thread it through**

`src/main/java/com/learnings/rag/retrieval/RetrievalOptions.java`:

```java
package com.learnings.rag.retrieval;

import com.learnings.rag.config.RagProperties;

/**
 * Per-call retrieval settings. The defaults come from {@code rag.retrieval.*}; the eval harness overrides them, so
 * one run can compare several configurations and retrieve deeper than the chat endpoint does.
 *
 * @param topK number of chunks to return
 * @param similarityThreshold minimum cosine similarity; 0 keeps everything
 */
public record RetrievalOptions(int topK, double similarityThreshold) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
    }

    public static RetrievalOptions from(RagProperties properties) {
        return new RetrievalOptions(properties.retrieval().topK(), properties.retrieval().similarityThreshold());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold);
    }
}
```

Replace `src/main/java/com/learnings/rag/retrieval/VectorRetriever.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.filter.FilterExpressionBuilder;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;

/**
 * Dense retrieval: cosine similarity over pgvector's HNSW index. Each Document's score is 1 - cosine distance.
 * Only chunks embedded by the configured model are searched: vectors from another model (an upload from before a
 * model switch, or the not-yet-re-ingested part of the corpus) are not comparable with the query vector.
 */
@Component
public class VectorRetriever implements DocumentRetriever {

    private final VectorStore vectorStore;
    private final RetrievalOptions defaults;
    private final Filter.Expression currentModel;

    public VectorRetriever(VectorStore vectorStore, RagProperties properties) {
        this.vectorStore = vectorStore;
        this.defaults = RetrievalOptions.from(properties);
        this.currentModel = new FilterExpressionBuilder()
                .eq(ChunkMetadata.EMBEDDING_MODEL, properties.embeddingModel())
                .build();
    }

    @Override
    public List<Document> retrieve(Query query) {
        return retrieve(query, defaults);
    }

    public List<Document> retrieve(Query query, RetrievalOptions options) {
        return vectorStore.similaritySearch(SearchRequest.builder()
                .query(query.text())
                .topK(options.topK())
                .similarityThreshold(options.similarityThreshold())
                .filterExpression(currentModel)
                .build());
    }
}
```

Replace `src/main/java/com/learnings/rag/retrieval/RetrievalPipeline.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.List;
import java.util.concurrent.TimeUnit;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;

/**
 * M2 baseline: a single vector search. Later milestones add query rewriting, multi-query expansion,
 * keyword search, rank fusion and reranking here, each as a traced stage.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.defaults = RetrievalOptions.from(properties);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Document> documents = vectorRetriever.retrieve(new Query(question), options);
        long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

        PipelineTrace.Stage vector = new PipelineTrace.Stage("vector", elapsed,
                documents.stream().map(PipelineTrace.Hit::of).toList());
        return new RetrievalResult(documents, new PipelineTrace(List.of(vector)));
    }
}
```

- [ ] **Step 5: Run the tests and watch them pass**

Run: `./mvnw -q verify -Dtest='RetrievalOptionsTest,AnswerServiceTest' -Dit.test=RetrievalPipelineIT`
Expected: PASS (3 + 5 unit tests, 4 ITs).

- [ ] **Step 6: Commit**

```bash
git add src/main/java/com/learnings/rag/retrieval src/test/java/com/learnings/rag/retrieval
git commit -m "feat: per-call retrieval options so the eval can retrieve deeper than chat"
```

---

### Task 2: Golden set model and file (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/{EvalProperties, ExpectedSource, GoldenItem, GoldenSet, InvalidGoldenSetException, GoldenSetFile}.java`
- Modify: `src/main/resources/application.yml` (add the `rag.eval` block)
- Test: `src/test/java/com/learnings/rag/eval/ExpectedSourceTest.java`, `src/test/java/com/learnings/rag/eval/GoldenSetFileTest.java`

**Interfaces:**
- Produces:
  - `EvalProperties`:
    - `record EvalProperties(Path goldenSet, Path reportsDir, Golden golden)` bound to `rag.eval`;
    - `record EvalProperties.Golden(Path draft, int size, long seed, int minTokens, int attemptsPerPage, int maxSharedWords, boolean overwrite)`.
  - `record ExpectedSource(String sourcePath, String sectionPrefix)` with `boolean matches(String chunkSourcePath, String chunkBreadcrumb)`.
  - `record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer, String sourceExcerpt)`.
  - `record GoldenSet(Path path, String sha256, List<GoldenItem> items)`.
  - `class InvalidGoldenSetException extends RuntimeException`.
  - `@Component GoldenSetFile(JsonMapper)`:
    - `GoldenSet read(Path)` validates and throws `InvalidGoldenSetException` listing every problem;
    - `void writeDraft(Path, List<GoldenItem>, boolean overwrite)`;
    - `void ensureDraftWritable(Path, boolean overwrite)` throws `IllegalStateException`.

- [ ] **Step 1: Write the failing tests**

`src/test/java/com/learnings/rag/eval/ExpectedSourceTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class ExpectedSourceTest {

    private final ExpectedSource indexes = new ExpectedSource("pgvector.adoc", "Indexes");

    @Test
    void matchesTheSectionItselfAndItsSubsections() {
        assertThat(indexes.matches("pgvector.adoc", "Indexes")).isTrue();
        assertThat(indexes.matches("pgvector.adoc", "Indexes › HNSW")).isTrue();
    }

    @Test
    void respectsHeadingBoundaries() {
        assertThat(new ExpectedSource("pgvector.adoc", "Index").matches("pgvector.adoc", "Indexes › HNSW")).isFalse();
        assertThat(indexes.matches("pgvector.adoc", "Auto-Configuration › Indexes")).isFalse();
    }

    @Test
    void otherPagesNeverMatch() {
        assertThat(indexes.matches("qdrant.adoc", "Indexes")).isFalse();
    }

    @Test
    void emptyPrefixAcceptsAnyChunkOfThePage() {
        ExpectedSource page = new ExpectedSource("pgvector.adoc", "");

        assertThat(page.matches("pgvector.adoc", "")).isTrue();
        assertThat(page.matches("pgvector.adoc", "Indexes › HNSW")).isTrue();
    }

    @Test
    void aParentSectionChunkDoesNotCountForADeeperExpectation() {
        assertThat(new ExpectedSource("pgvector.adoc", "Indexes › HNSW").matches("pgvector.adoc", "Indexes")).isFalse();
    }
}
```

`src/test/java/com/learnings/rag/eval/GoldenSetFileTest.java`:

```java
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
```

- [ ] **Step 2: Run the tests and watch them fail**

Run: `./mvnw -q test -Dtest='ExpectedSourceTest,GoldenSetFileTest'`
Expected: compilation FAILURE, `cannot find symbol: class ExpectedSource`.

- [ ] **Step 3: Implement the golden set model and file**

`src/main/java/com/learnings/rag/eval/EvalProperties.java`:

```java
package com.learnings.rag.eval;

import java.nio.file.Path;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.bind.DefaultValue;

/**
 * @param goldenSet the reviewed golden set the eval scores against (committed)
 * @param reportsDir where eval reports are written (gitignored)
 * @param golden how the golden-set generator samples and writes its draft
 */
@ConfigurationProperties("rag.eval")
public record EvalProperties(@DefaultValue("eval/golden-set.json") Path goldenSet,
        @DefaultValue("eval/reports") Path reportsDir,
        @DefaultValue Golden golden) {

    /**
     * @param draft where the generator writes questions for review (never the reviewed golden set itself)
     * @param size questions to generate: more than the ~30 kept, because review removes weak ones
     * @param seed makes page and chunk sampling repeatable
     * @param minTokens chunks smaller than this are too thin to ask about
     * @param attemptsPerPage chunks tried per page in one round before moving on to the next page
     * @param maxSharedWords a question sharing this many consecutive words with its chunk is rewritten once,
     *        then dropped
     * @param overwrite replace an existing draft; otherwise the generator refuses, to protect review edits
     */
    public record Golden(@DefaultValue("eval/golden-set.draft.json") Path draft,
            @DefaultValue("40") int size,
            @DefaultValue("42") long seed,
            @DefaultValue("80") int minTokens,
            @DefaultValue("3") int attemptsPerPage,
            @DefaultValue("5") int maxSharedWords,
            @DefaultValue("false") boolean overwrite) {
    }
}
```

`src/main/java/com/learnings/rag/eval/ExpectedSource.java`:

```java
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
```

`src/main/java/com/learnings/rag/eval/GoldenItem.java`:

```java
package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section needed to answer; hit@5 needs one of them, recall@5 counts how many
 * @param referenceAnswer a short answer written from the source chunk; used by the answer evals in M7
 * @param sourceExcerpt the chunk the question was generated from, as context for reviewers; never scored
 */
public record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
        String sourceExcerpt) {
}
```

`src/main/java/com/learnings/rag/eval/GoldenSet.java`:

```java
package com.learnings.rag.eval;

import java.nio.file.Path;
import java.util.List;

/** A validated golden set. @param sha256 of the file, recorded in every report so results stay traceable */
public record GoldenSet(Path path, String sha256, List<GoldenItem> items) {
}
```

`src/main/java/com/learnings/rag/eval/InvalidGoldenSetException.java`:

```java
package com.learnings.rag.eval;

/** The golden set file is missing, unreadable, or has items that cannot be scored. */
public class InvalidGoldenSetException extends RuntimeException {

    public InvalidGoldenSetException(String message) {
        super(message);
    }
}
```

`src/main/java/com/learnings/rag/eval/GoldenSetFile.java`:

```java
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
```

In `src/main/resources/application.yml`, add this under the existing `rag:` block, after `retrieval:` and at the same indentation as `retrieval:`:

```yaml
  eval:
    golden-set: eval/golden-set.json         # reviewed by a human, committed
    reports-dir: eval/reports                # gitignored
    golden:
      draft: eval/golden-set.draft.json      # gitignored; review it, then save as golden-set.json
      size: 40
      seed: 42
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./mvnw -q test -Dtest='ExpectedSourceTest,GoldenSetFileTest'`
Expected: PASS (5 + 6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval src/main/resources/application.yml
git commit -m "feat: golden set model, heading-boundary matching, validated loading and safe draft writing"
```

---

### Task 3: Retrieval metrics (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/RetrievalMetrics.java`
- Test: `src/test/java/com/learnings/rag/eval/RetrievalMetricsTest.java`

**Interfaces:**
- Consumes: `ExpectedSource.matches` (Task 2).
- Produces:
  - `final class RetrievalMetrics`:
    - constants `HIT_K = 5` and `MRR_K = 10`;
    - `static ItemScore score(List<ExpectedSource>, List<RankedSource>)`;
    - `static Summary summarize(List<ItemScore>, List<Long> latenciesMillis)`;
    - `static long percentile(List<Long>, double p)`.
  - Nested records:
    - `RankedSource(String sourcePath, String breadcrumb, Double score)`;
    - `ItemScore(Integer firstRelevantRank, boolean hitAt5, double recallAt5, double reciprocalRankAt10)`;
    - `Summary(int items, double hitAt5, double recallAt5, double mrrAt10, long p50Millis, long p95Millis)`.

- [ ] **Step 1: Write the failing test**

`src/test/java/com/learnings/rag/eval/RetrievalMetricsTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;

import com.learnings.rag.eval.RetrievalMetrics.ItemScore;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.eval.RetrievalMetrics.Summary;

class RetrievalMetricsTest {

    private static final ExpectedSource HNSW = new ExpectedSource("pgvector.adoc", "Indexes › HNSW");
    private static final ExpectedSource STREAMING = new ExpectedSource("chat-client.adoc", "Streaming");

    private static RankedSource chunk(String path, String breadcrumb) {
        return new RankedSource(path, breadcrumb, 0.5);
    }

    private static List<RankedSource> misses(int count) {
        return IntStream.range(0, count).mapToObj(i -> chunk("other.adoc", "Section " + i)).toList();
    }

    private static List<RankedSource> ranked(List<RankedSource> before, RankedSource relevant, int after) {
        List<RankedSource> ranked = new ArrayList<>(before);
        ranked.add(relevant);
        ranked.addAll(misses(after));
        return ranked;
    }

    @Test
    void relevantChunkAtRankThreeIsAHitWithReciprocalRankOneThird() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW),
                ranked(misses(2), chunk("pgvector.adoc", "Indexes › HNSW › Tuning"), 7));

        assertThat(score.firstRelevantRank()).isEqualTo(3);
        assertThat(score.hitAt5()).isTrue();
        assertThat(score.recallAt5()).isEqualTo(1.0);
        assertThat(score.reciprocalRankAt10()).isCloseTo(1.0 / 3, within(1e-9));
    }

    @Test
    void relevantChunkAtRankSevenCountsForMrrButNotForHitOrRecall() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW), ranked(misses(6), chunk("pgvector.adoc", "Indexes › HNSW"), 3));

        assertThat(score.firstRelevantRank()).isEqualTo(7);
        assertThat(score.hitAt5()).isFalse();
        assertThat(score.recallAt5()).isZero();
        assertThat(score.reciprocalRankAt10()).isCloseTo(1.0 / 7, within(1e-9));
    }

    @Test
    void nothingRelevantScoresZero() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW), misses(10));

        assertThat(score).isEqualTo(new ItemScore(null, false, 0.0, 0.0));
    }

    @Test
    void recallCountsEachExpectedSourceCoveredInTheTopFive() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW, STREAMING),
                ranked(List.of(), chunk("pgvector.adoc", "Indexes › HNSW"), 9));

        assertThat(score.firstRelevantRank()).isEqualTo(1);
        assertThat(score.recallAt5()).isEqualTo(0.5);
    }

    @Test
    void anyExpectedSourceCanProvideTheFirstHit() {
        ItemScore score = RetrievalMetrics.score(List.of(HNSW, STREAMING),
                ranked(misses(1), chunk("chat-client.adoc", "Streaming"), 8));

        assertThat(score.firstRelevantRank()).isEqualTo(2);
        assertThat(score.hitAt5()).isTrue();
    }

    @Test
    void summaryAveragesItemsAndTakesNearestRankLatencyPercentiles() {
        List<ItemScore> scores = List.of(
                new ItemScore(1, true, 1.0, 1.0),
                new ItemScore(4, true, 0.5, 0.25),
                new ItemScore(null, false, 0.0, 0.0),
                new ItemScore(2, true, 1.0, 0.5));

        Summary summary = RetrievalMetrics.summarize(scores, List.of(120L, 80L, 400L, 100L));

        assertThat(summary.items()).isEqualTo(4);
        assertThat(summary.hitAt5()).isEqualTo(0.75);
        assertThat(summary.recallAt5()).isEqualTo(0.625);
        assertThat(summary.mrrAt10()).isEqualTo(0.4375);
        assertThat(summary.p50Millis()).isEqualTo(100);
        assertThat(summary.p95Millis()).isEqualTo(400);
    }

    @Test
    void percentileOfASingleValueIsThatValue() {
        assertThat(RetrievalMetrics.percentile(List.of(42L), 0.95)).isEqualTo(42);
    }

    @Test
    void summarizingNothingIsAnError() {
        assertThatThrownBy(() -> RetrievalMetrics.summarize(List.of(), List.of()))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q test -Dtest=RetrievalMetricsTest`
Expected: compilation FAILURE, `cannot find symbol: class RetrievalMetrics`.

- [ ] **Step 3: Implement `RetrievalMetrics`**

`src/main/java/com/learnings/rag/eval/RetrievalMetrics.java`:

```java
package com.learnings.rag.eval;

import java.util.List;

/** Retrieval quality for one golden question, and averaged over a run. Pure functions. */
public final class RetrievalMetrics {

    /** hit@5 and recall@5 look at the first 5 chunks: what the chat endpoint hands the model. */
    public static final int HIT_K = 5;

    /** MRR@10 credits the first relevant chunk anywhere in the first 10. */
    public static final int MRR_K = 10;

    private RetrievalMetrics() {
    }

    /** One retrieved chunk as the metrics see it. @param score cosine similarity, for the report only */
    public record RankedSource(String sourcePath, String breadcrumb, Double score) {
    }

    /**
     * @param firstRelevantRank 1-based rank of the first chunk matching any expected source; null if none was
     *        retrieved
     * @param recallAt5 share of the expected sources matched within the first 5 chunks
     * @param reciprocalRankAt10 1 / firstRelevantRank when it is at most 10, otherwise 0
     */
    public record ItemScore(Integer firstRelevantRank, boolean hitAt5, double recallAt5, double reciprocalRankAt10) {
    }

    public record Summary(int items, double hitAt5, double recallAt5, double mrrAt10, long p50Millis,
            long p95Millis) {
    }

    public static ItemScore score(List<ExpectedSource> expected, List<RankedSource> ranked) {
        Integer firstRelevant = null;
        for (int i = 0; i < ranked.size() && firstRelevant == null; i++) {
            RankedSource chunk = ranked.get(i);
            if (expected.stream().anyMatch(source -> source.matches(chunk.sourcePath(), chunk.breadcrumb()))) {
                firstRelevant = i + 1;
            }
        }
        List<RankedSource> top = ranked.subList(0, Math.min(HIT_K, ranked.size()));
        long covered = expected.stream()
                .filter(source -> top.stream().anyMatch(chunk -> source.matches(chunk.sourcePath(), chunk.breadcrumb())))
                .count();
        boolean hit = firstRelevant != null && firstRelevant <= HIT_K;
        double reciprocalRank = firstRelevant != null && firstRelevant <= MRR_K ? 1.0 / firstRelevant : 0.0;
        return new ItemScore(firstRelevant, hit, (double) covered / expected.size(), reciprocalRank);
    }

    public static Summary summarize(List<ItemScore> scores, List<Long> latenciesMillis) {
        if (scores.isEmpty()) {
            throw new IllegalArgumentException("No scores to summarize");
        }
        return new Summary(scores.size(),
                scores.stream().mapToDouble(score -> score.hitAt5() ? 1 : 0).average().orElseThrow(),
                scores.stream().mapToDouble(ItemScore::recallAt5).average().orElseThrow(),
                scores.stream().mapToDouble(ItemScore::reciprocalRankAt10).average().orElseThrow(),
                percentile(latenciesMillis, 0.50),
                percentile(latenciesMillis, 0.95));
    }

    /** Nearest-rank percentile: the smallest value with at least {@code p} of all values at or below it. */
    public static long percentile(List<Long> values, double p) {
        if (values.isEmpty()) {
            throw new IllegalArgumentException("No values");
        }
        List<Long> sorted = values.stream().sorted().toList();
        int index = (int) Math.ceil(p * sorted.size()) - 1;
        return sorted.get(Math.clamp(index, 0, sorted.size() - 1));
    }
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `./mvnw -q test -Dtest=RetrievalMetricsTest`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval/RetrievalMetrics.java src/test/java/com/learnings/rag/eval/RetrievalMetricsTest.java
git commit -m "feat: retrieval metrics: hit@5, recall@5, MRR@10 and nearest-rank latency percentiles"
```

---

### Task 4: Corpus chunk catalog (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/{CorpusChunk, IndexStats, ChunkCatalog}.java`
- Test: `src/test/java/com/learnings/rag/eval/ChunkCatalogIT.java`

**Interfaces:**
- Consumes: `ChunkMetadata` constants; `RagProperties.embeddingModel()`; the `vector_store` and `source_document` tables.
- Produces:
  - `record CorpusChunk(String sourcePath, String title, String breadcrumb, int chunkIndex, int tokenCount, String content)`;
  - `record IndexStats(int corpusDocuments, int uploadedDocuments, int chunks)`;
  - `@Repository ChunkCatalog(JdbcClient, RagProperties)` with `List<CorpusChunk> corpusChunks()` (current model, corpus only, ordered by page then chunk index) and `IndexStats stats()` (current model).

- [ ] **Step 1: Write the failing integration test**

`src/test/java/com/learnings/rag/eval/ChunkCatalogIT.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.transaction.PlatformTransactionManager;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.ingest.CorpusIngestor;
import com.learnings.rag.ingest.DocumentIngestionService;
import com.learnings.rag.ingest.SourceDocument.Origin;
import com.learnings.rag.ingest.SourceDocumentRepository;
import com.learnings.rag.ingest.StructureAwareChunker;

@RagIntegrationTest
class ChunkCatalogIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    DocumentIngestionService ingestion;

    @Autowired
    ChunkCatalog catalog;

    @Autowired
    JdbcClient jdbc;

    @Autowired
    VectorStore vectorStore;

    @Autowired
    SourceDocumentRepository documents;

    @Autowired
    RagProperties properties;

    @Autowired
    PlatformTransactionManager transactionManager;

    @TempDir
    Path corpus;

    @BeforeEach
    void ingest() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        for (String name : new String[] { "pgvector.adoc", "chat-client.adoc" }) {
            try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
                Files.copy(in, corpus.resolve(name));
            }
        }
        corpusIngestor.ingestDirectory(corpus);
        ingestion.ingest("uploads/notes.md", "notes", "# Notes\n\nUploaded notes about HNSW tuning.", Origin.UPLOAD);
        RagProperties retiredModel = new RagProperties(properties.corpusDir(), "retired-embedding-model",
                properties.chunking(), properties.retrieval());
        new DocumentIngestionService(vectorStore, documents, new StructureAwareChunker(properties), retiredModel,
                transactionManager)
                .ingest("old.adoc", "old", "= Old\n\nEmbedded with a retired model.", Origin.CORPUS);
    }

    @Test
    void corpusChunksAreTheCurrentModelsCorpusChunksInPageOrder() {
        List<CorpusChunk> chunks = catalog.corpusChunks();

        assertThat(chunks).extracting(CorpusChunk::sourcePath)
                .containsOnly("chat-client.adoc", "pgvector.adoc")
                .isSorted();
        assertThat(chunks).filteredOn(chunk -> chunk.sourcePath().equals("pgvector.adoc"))
                .extracting(CorpusChunk::chunkIndex)
                .isSorted();
        assertThat(chunks).anySatisfy(chunk -> {
            assertThat(chunk.title()).isEqualTo("PGvector");
            assertThat(chunk.breadcrumb()).isEqualTo("Configuration properties › HNSW index");
            assertThat(chunk.content()).startsWith("PGvector › Configuration properties › HNSW index\n\n");
            assertThat(chunk.tokenCount()).isPositive();
        });
    }

    @Test
    void statsCountCorpusPagesUploadsAndChunksForTheCurrentModel() {
        int currentModelChunks = jdbc.sql("SELECT count(*)::int FROM vector_store WHERE metadata->>'"
                + ChunkMetadata.EMBEDDING_MODEL + "' = :model")
                .param("model", properties.embeddingModel())
                .query(Integer.class).single();

        assertThat(catalog.stats()).isEqualTo(new IndexStats(2, 1, currentModelChunks));
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q verify -Dit.test=ChunkCatalogIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false`
Expected: compilation FAILURE, `cannot find symbol: class ChunkCatalog`.

- [ ] **Step 3: Implement the catalog**

`src/main/java/com/learnings/rag/eval/CorpusChunk.java`:

```java
package com.learnings.rag.eval;

/**
 * A stored corpus chunk, as the golden-set generator samples it.
 *
 * @param content the contextual header plus body, exactly as embedded
 */
public record CorpusChunk(String sourcePath, String title, String breadcrumb, int chunkIndex, int tokenCount,
        String content) {
}
```

`src/main/java/com/learnings/rag/eval/IndexStats.java`:

```java
package com.learnings.rag.eval;

/** What the index holds for the current embedding model; recorded in every eval report. */
public record IndexStats(int corpusDocuments, int uploadedDocuments, int chunks) {
}
```

`src/main/java/com/learnings/rag/eval/ChunkCatalog.java`:

```java
package com.learnings.rag.eval;

import static com.learnings.rag.ingest.ChunkMetadata.BREADCRUMB;
import static com.learnings.rag.ingest.ChunkMetadata.CHUNK_INDEX;
import static com.learnings.rag.ingest.ChunkMetadata.EMBEDDING_MODEL;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_ID;
import static com.learnings.rag.ingest.ChunkMetadata.SOURCE_PATH;
import static com.learnings.rag.ingest.ChunkMetadata.TITLE;
import static com.learnings.rag.ingest.ChunkMetadata.TOKEN_COUNT;

import java.util.List;

import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Repository;

import com.learnings.rag.config.RagProperties;

/** Read-only views of the index for the eval harness. Only chunks of the current embedding model count. */
@Repository
public class ChunkCatalog {

    private static final String CORPUS_CHUNKS = """
            SELECT v.metadata->>'%s' AS source_path,
                   v.metadata->>'%s' AS title,
                   v.metadata->>'%s' AS breadcrumb,
                   (v.metadata->>'%s')::int AS chunk_index,
                   (v.metadata->>'%s')::int AS token_count,
                   v.content
            FROM vector_store v
            JOIN source_document d ON d.id::text = v.metadata->>'%s'
            WHERE d.origin = 'CORPUS' AND v.metadata->>'%s' = :model
            ORDER BY source_path, chunk_index
            """.formatted(SOURCE_PATH, TITLE, BREADCRUMB, CHUNK_INDEX, TOKEN_COUNT, SOURCE_ID, EMBEDDING_MODEL);

    private static final String STATS = """
            SELECT count(DISTINCT d.id) FILTER (WHERE d.origin = 'CORPUS') AS corpus_documents,
                   count(DISTINCT d.id) FILTER (WHERE d.origin = 'UPLOAD') AS uploaded_documents,
                   count(*) AS chunks
            FROM vector_store v
            JOIN source_document d ON d.id::text = v.metadata->>'%s'
            WHERE v.metadata->>'%s' = :model
            """.formatted(SOURCE_ID, EMBEDDING_MODEL);

    private final JdbcClient jdbc;
    private final String embeddingModel;

    public ChunkCatalog(JdbcClient jdbc, RagProperties properties) {
        this.jdbc = jdbc;
        this.embeddingModel = properties.embeddingModel();
    }

    /** Corpus chunks (uploads excluded), ordered by page and then by position in the page. */
    public List<CorpusChunk> corpusChunks() {
        return jdbc.sql(CORPUS_CHUNKS)
                .param("model", embeddingModel)
                .query((rs, rowNum) -> new CorpusChunk(
                        rs.getString("source_path"),
                        rs.getString("title"),
                        rs.getString("breadcrumb"),
                        rs.getInt("chunk_index"),
                        rs.getInt("token_count"),
                        rs.getString("content")))
                .list();
    }

    public IndexStats stats() {
        return jdbc.sql(STATS)
                .param("model", embeddingModel)
                .query((rs, rowNum) -> new IndexStats(
                        rs.getInt("corpus_documents"),
                        rs.getInt("uploaded_documents"),
                        rs.getInt("chunks")))
                .single();
    }
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `./mvnw -q verify -Dit.test=ChunkCatalogIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval/ChunkCatalogIT.java
git commit -m "feat: chunk catalog: corpus chunks and index composition for the current embedding model"
```

---

### Task 5: Golden set generator (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/{LexicalOverlap, GeneratedQuestion, QuestionWriter, GoldenSetGenerator}.java`
- Test: `src/test/java/com/learnings/rag/eval/LexicalOverlapTest.java`, `src/test/java/com/learnings/rag/eval/GoldenSetGeneratorTest.java`

**Interfaces:**
- Consumes: `CorpusChunk` (Task 4); `EvalProperties.Golden` (Task 2); `GoldenItem` and `ExpectedSource` (Task 2).
- Produces:
  - `final class LexicalOverlap` (package-private) with `static String longestSharedRun(String a, String b)`, `static List<String> words(String)` and `static int wordCount(String)`.
  - `record GeneratedQuestion(boolean usable, String question, String referenceAnswer)`.
  - `interface QuestionWriter { GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase); }`. `rejectedPhrase` is null on the first attempt.
  - `@Service GoldenSetGenerator`:
    - constructors `(QuestionWriter, EvalProperties)` (`@Autowired`) and `(QuestionWriter, EvalProperties.Golden)` (package-private, for tests);
    - `List<GoldenItem> generate(List<CorpusChunk>)`;
    - `static final int MAX_CONSECUTIVE_FAILURES = 3`.

- [ ] **Step 1: Write the failing tests**

`src/test/java/com/learnings/rag/eval/LexicalOverlapTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class LexicalOverlapTest {

    @Test
    void findsACopiedPhraseIgnoringCaseAndPunctuation() {
        String chunk = "It builds a multilayer graph. It has better query performance than IVFFlat but slower build times.";

        assertThat(LexicalOverlap.longestSharedRun(
                "Does HNSW give Better Query Performance than IVFFlat?", chunk))
                .isEqualTo("better query performance than ivfflat");
    }

    @Test
    void ownWordsShareOnlyShortRuns() {
        assertThat(LexicalOverlap.wordCount(LexicalOverlap.longestSharedRun(
                "Which setting turns on the multilayer graph index?",
                "The HNSW index type builds a multilayer graph. It uses more memory.")))
                .isEqualTo(2);
    }

    @Test
    void identifiersCountAsOneWord() {
        assertThat(LexicalOverlap.words("Set `spring.ai.vectorstore.pgvector.index-type` on ChatClient.Builder, then build."))
                .containsExactly("set", "spring.ai.vectorstore.pgvector.index-type", "on", "chatclient.builder", "then",
                        "build");
    }

    @Test
    void noSharedWordsGiveAnEmptyRun() {
        assertThat(LexicalOverlap.longestSharedRun("Why?", "Because.")).isEmpty();
    }
}
```

`src/test/java/com/learnings/rag/eval/GoldenSetGeneratorTest.java`:

```java
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
```

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest='LexicalOverlapTest,GoldenSetGeneratorTest'`
Expected: compilation FAILURE, `cannot find symbol: class LexicalOverlap`.

- [ ] **Step 3: Implement the generator**

`src/main/java/com/learnings/rag/eval/LexicalOverlap.java`:

```java
package com.learnings.rag.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Detects a question that copies wording from its source chunk. Copied phrases make a question unrealistically easy
 * for keyword search, which would bias every comparison the golden set is used for. Identifiers
 * ({@code spring.ai.vectorstore.pgvector.index-type}, {@code ChatClient.Builder}) count as one word: naming the thing
 * you ask about is fine; copying the sentence around it is not.
 */
final class LexicalOverlap {

    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+(?:[._\\-/:#][\\p{L}\\p{N}]+)*");

    private LexicalOverlap() {
    }

    /** The longest run of consecutive words found in both texts (case-insensitive), or "" when none. */
    static String longestSharedRun(String a, String b) {
        List<String> x = words(a);
        List<String> y = words(b);
        int best = 0;
        int end = 0;
        int[] previous = new int[y.size() + 1];
        for (int i = 1; i <= x.size(); i++) {
            int[] current = new int[y.size() + 1];
            for (int j = 1; j <= y.size(); j++) {
                if (x.get(i - 1).equals(y.get(j - 1))) {
                    current[j] = previous[j - 1] + 1;
                    if (current[j] > best) {
                        best = current[j];
                        end = i;
                    }
                }
            }
            previous = current;
        }
        return String.join(" ", x.subList(end - best, end));
    }

    static List<String> words(String text) {
        List<String> words = new ArrayList<>();
        Matcher matcher = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            words.add(matcher.group());
        }
        return words;
    }

    static int wordCount(String phrase) {
        return words(phrase).size();
    }
}
```

`src/main/java/com/learnings/rag/eval/GeneratedQuestion.java`:

```java
package com.learnings.rag.eval;

/**
 * What the question writer returns for one chunk.
 *
 * @param usable false when the chunk is not worth a question (mostly code, links, a version table, boilerplate)
 */
public record GeneratedQuestion(boolean usable, String question, String referenceAnswer) {
}
```

`src/main/java/com/learnings/rag/eval/QuestionWriter.java`:

```java
package com.learnings.rag.eval;

/** Writes one evaluation question about a chunk. The production implementation is {@link LlmQuestionWriter}. */
public interface QuestionWriter {

    /**
     * @param rejectedPhrase a phrase the previous attempt copied from the chunk and must avoid; null on the first
     *        attempt
     */
    GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase);
}
```

`src/main/java/com/learnings/rag/eval/GoldenSetGenerator.java`:

```java
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
```

- [ ] **Step 4: Run them and watch them pass**

Run: `./mvnw -q test -Dtest='LexicalOverlapTest,GoldenSetGeneratorTest'`
Expected: PASS (4 + 11 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval
git commit -m "feat: golden set generator: even sampling across pages, leakage guard, fail-fast on LLM outages"
```

---

### Task 6: LLM question writer (M3)

**Files:**
- Create: `src/main/resources/prompts/golden-question.st`, `src/main/java/com/learnings/rag/eval/LlmQuestionWriter.java`
- Test: `src/test/java/com/learnings/rag/eval/LlmQuestionWriterTest.java`

**Interfaces:**
- Consumes: `QuestionWriter`, `GeneratedQuestion`, `CorpusChunk` (Tasks 4–5); `ChatClient.Builder` (Spring AI auto-config); `StubChatModel(String...)` and `prompts()` (test support from M2).
- Produces: `@Component LlmQuestionWriter(ChatClient.Builder, Resource systemPrompt) implements QuestionWriter`, plus `static String userMessage(CorpusChunk, String rejectedPhrase)`.

- [ ] **Step 1: Write the system prompt `src/main/resources/prompts/golden-question.st`**

```
You write evaluation questions for a search system over the Spring AI reference documentation.

You get one excerpt from the documentation. Write ONE question that a Java developer using Spring AI would realistically type into a documentation search box, and that this excerpt answers.

Rules:
- The excerpt must contain the answer. Never ask about something it does not state.
- Use your own words. Do not copy phrases from the excerpt: no run of four or more consecutive words from it. Class, method, annotation and property names may be used as they are.
- Ask about one thing. No compound questions.
- Do not refer to "the excerpt", "this section", "this page" or "the documentation": the developer has not seen it.
- Phrase it the way a developer with a goal would: "How do I…", "Why does…", "What happens if…", "Which property…".
- referenceAnswer: one to three sentences that answer the question using only the excerpt.
- If the excerpt is mostly code, a list of links, a version or compatibility table, or boilerplate nobody would search for, set usable to false and leave question and referenceAnswer empty.
```

- [ ] **Step 2: Write the failing test**

`src/test/java/com/learnings/rag/eval/LlmQuestionWriterTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.core.io.ClassPathResource;

import com.learnings.rag.StubChatModel;

class LlmQuestionWriterTest {

    private static final CorpusChunk CHUNK = new CorpusChunk("api/prompt.adoc", "Prompts", "Templates", 3, 120,
            "Prompts › Templates\n\nUse {name} placeholders; ${user.home} is not expanded.");

    private static LlmQuestionWriter writer(StubChatModel model) {
        return new LlmQuestionWriter(ChatClient.builder(model), new ClassPathResource("prompts/golden-question.st"));
    }

    @Test
    void parsesTheStructuredReply() {
        StubChatModel model = new StubChatModel(
                "{\"usable\": true, \"question\": \"How do I fill a placeholder?\", \"referenceAnswer\": \"Pass a value.\"}");

        assertThat(writer(model).write(CHUNK, null))
                .isEqualTo(new GeneratedQuestion(true, "How do I fill a placeholder?", "Pass a value."));
    }

    @Test
    void promptCarriesTheRulesTheChunkVerbatimAndTheJsonFormat() {
        StubChatModel model = new StubChatModel("{\"usable\": false, \"question\": \"\", \"referenceAnswer\": \"\"}");

        writer(model).write(CHUNK, null);

        assertThat(model.prompts()).singleElement().satisfies(prompt -> {
            assertThat(prompt.getSystemMessage().getText()).contains("Use your own words", "usable to false");
            assertThat(prompt.getUserMessage().getText())
                    .contains("Document: Prompts", "Section: Templates",
                            "Use {name} placeholders; ${user.home} is not expanded.")
                    .contains("JSON");
        });
    }

    @Test
    void aRetryNamesThePhraseToAvoid() {
        assertThat(LlmQuestionWriter.userMessage(CHUNK, "use name placeholders"))
                .endsWith("Your previous question copied this phrase from the excerpt: \"use name placeholders\". "
                        + "Write a different question in your own words.");
    }

    @Test
    void introductionChunksAreLabelled() {
        CorpusChunk intro = new CorpusChunk("api/prompt.adoc", "Prompts", "", 0, 90, "Prompts\n\nPrompts guide models.");

        assertThat(LlmQuestionWriter.userMessage(intro, null)).contains("Section: (introduction)");
    }
}
```

- [ ] **Step 3: Run it and watch it fail**

Run: `./mvnw -q test -Dtest=LlmQuestionWriterTest`
Expected: compilation FAILURE, `cannot find symbol: class LlmQuestionWriter`.

- [ ] **Step 4: Implement `LlmQuestionWriter`**

`src/main/java/com/learnings/rag/eval/LlmQuestionWriter.java`:

```java
package com.learnings.rag.eval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.List;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

/** Asks the chat model for one question per chunk, as structured output ({@link GeneratedQuestion}). */
@Component
public class LlmQuestionWriter implements QuestionWriter {

    private final ChatClient chatClient;
    private final String systemPrompt;

    public LlmQuestionWriter(ChatClient.Builder chatClientBuilder,
            @Value("classpath:prompts/golden-question.st") Resource systemPrompt) {
        this.chatClient = chatClientBuilder.build();
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public GeneratedQuestion write(CorpusChunk chunk, String rejectedPhrase) {
        // Message objects, not template strings: chunks are full of {braces} from code samples.
        return chatClient
                .prompt(new Prompt(List.of(new SystemMessage(systemPrompt),
                        new UserMessage(userMessage(chunk, rejectedPhrase)))))
                .call()
                .entity(GeneratedQuestion.class);
    }

    static String userMessage(CorpusChunk chunk, String rejectedPhrase) {
        StringBuilder text = new StringBuilder()
                .append("Document: ").append(chunk.title()).append('\n')
                .append("Section: ").append(chunk.breadcrumb().isEmpty() ? "(introduction)" : chunk.breadcrumb())
                .append("\n\n<excerpt>\n").append(chunk.content()).append("\n</excerpt>");
        if (rejectedPhrase != null) {
            text.append("\n\nYour previous question copied this phrase from the excerpt: \"").append(rejectedPhrase)
                    .append("\". Write a different question in your own words.");
        }
        return text.toString();
    }
}
```

- [ ] **Step 5: Run it and watch it pass**

Run: `./mvnw -q test -Dtest=LlmQuestionWriterTest`
Expected: PASS (4 tests). The second test proves that `{name}` and `${user.home}` reach the model verbatim and that Spring AI appended its JSON format instructions.

- [ ] **Step 6: Run every integration test**

Spring now creates `LlmQuestionWriter` and `GoldenSetGenerator` in every IT context (from the `ChatClient.Builder` over `StubChatModel`).
Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS.

- [ ] **Step 7: Commit**

```bash
git add src/main/resources/prompts/golden-question.st src/main/java/com/learnings/rag/eval/LlmQuestionWriter.java src/test/java/com/learnings/rag/eval/LlmQuestionWriterTest.java
git commit -m "feat: LLM question writer with structured output and a no-copying prompt"
```

---

### Task 7: Eval runner and reports (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/{EvalConfig, EvalReport, EvalRunner, ReportWriter}.java`
- Test: `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`, `src/test/java/com/learnings/rag/eval/EvalRunnerIT.java`

**Interfaces:**
- Consumes:
  - `RetrievalPipeline.retrieve(String, RetrievalOptions)`, `RetrievalOptions` (Task 1);
  - `GoldenSet`, `GoldenItem`, `ExpectedSource` (Task 2);
  - `RetrievalMetrics` (Task 3);
  - `ChunkCatalog.stats()`, `IndexStats` (Task 4);
  - `SourceDocumentRepository.findBySourcePath`; `ChunkMetadata.SOURCE_PATH` and `BREADCRUMB`.
- Produces:
  - `record EvalConfig(String name, RetrievalOptions options)` with `static List<EvalConfig> all(RagProperties)`. In M3 that is `vector` with top 10.
  - `record EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs)` with nested records:
    - `RunInfo(String goldenSet, String goldenSetSha256, int goldenItems, String embeddingModel, RagProperties.Chunking chunking, IndexStats index)`;
    - `ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary, List<ItemResult> items)`;
    - `ItemResult(String id, String question, List<ExpectedSource> expectedSources, RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved)`.
  - `@Service EvalRunner` with `EvalReport run(GoldenSet, List<EvalConfig>)` (throws `IllegalStateException` on an empty index or unindexed sources) and `List<String> unindexedSources(GoldenSet)`.
  - `@Component ReportWriter(JsonMapper)` with `Path write(EvalReport, Path directory)` and `static String markdown(EvalReport)`.

- [ ] **Step 1: Write the failing tests**

`src/test/java/com/learnings/rag/eval/ReportWriterTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.time.Instant;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.RetrievalMetrics.ItemScore;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.retrieval.RetrievalOptions;

import tools.jackson.databind.json.JsonMapper;

class ReportWriterTest {

    @TempDir
    Path dir;

    private static EvalReport report(int uploads) {
        ItemResult hit = new ItemResult("q01", "Which index | type builds a graph?",
                List.of(new ExpectedSource("api/vectordbs/pgvector.adoc", "Indexes")),
                new ItemScore(1, true, 1.0, 1.0), 120,
                List.of(new RankedSource("api/vectordbs/pgvector.adoc", "Indexes › HNSW", 0.61)));
        ItemResult miss = new ItemResult("q02", "How do I\nstream tokens?",
                List.of(new ExpectedSource("api/chatclient.adoc", "Streaming")),
                new ItemScore(null, false, 0.0, 0.0), 80,
                List.of(new RankedSource("upgrade-notes.adoc", "Upgrading to 1.0.0-M8", 0.2)));
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(List.of(hit.score(), miss.score()),
                List.of(120L, 80L));
        RunInfo run = new RunInfo("eval/golden-set.json", "a".repeat(64), 2, "text-embedding-3-small",
                new RagProperties.Chunking(500, 50, 60), new IndexStats(52, uploads, 1106));
        return new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), run,
                List.of(new ConfigResult("vector", new RetrievalOptions(10, 0.0), summary, List.of(hit, miss))));
    }

    @Test
    void writesMarkdownAndJsonNamedAfterTheStartTime() {
        Path markdown = new ReportWriter(JsonMapper.builder().build()).write(report(0), dir.resolve("reports"));

        assertThat(markdown).hasFileName("2026-10-06T09-30-00Z.md").exists();
        assertThat(dir.resolve("reports/2026-10-06T09-30-00Z.json")).content()
                .contains("\"hitAt5\"", "text-embedding-3-small", "Upgrading to 1.0.0-M8");
    }

    @Test
    void summaryRowShowsMetricsToThreeDecimalsAndLatencyPercentiles() {
        assertThat(ReportWriter.markdown(report(0))).contains("| vector | 0.500 | 0.500 | 0.500 | 80 | 120 |");
    }

    @Test
    void missesListWhatWasExpectedAndWhatCameBack() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "- **q02** How do I stream tokens?",
                "expected: `api/chatclient.adoc` › Streaming",
                "retrieved: `upgrade-notes.adoc` › Upgrading to 1.0.0-M8");
    }

    @Test
    void questionsCannotBreakTheTable() {
        assertThat(ReportWriter.markdown(report(0))).contains("| Which index \\| type builds a graph? |");
    }

    @Test
    void warnsWhenUploadsShareTheIndex() {
        assertThat(ReportWriter.markdown(report(2))).contains("2 uploaded document(s)");
        assertThat(ReportWriter.markdown(report(0))).doesNotContain("uploaded document");
    }

    @Test
    void headerRecordsWhatWasMeasured() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "`eval/golden-set.json` (2 questions, sha256 `aaaaaaaaaaaa…`)",
                "52 corpus pages, 0 uploads, 1106 chunks",
                "`text-embedding-3-small`",
                "max 500 · min 50 · overlap 60 tokens");
    }
}
```

`src/test/java/com/learnings/rag/eval/EvalRunnerIT.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.ingest.CorpusIngestor;

@RagIntegrationTest
class EvalRunnerIT {

    private static final String HNSW_QUESTION = "Which index type is HNSW and how does it build its graph?";

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    EvalRunner runner;

    @Autowired
    RagProperties properties;

    @Autowired
    JdbcClient jdbc;

    @TempDir
    Path corpus;

    @BeforeEach
    void ingestFixtures() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        for (String name : new String[] { "pgvector.adoc", "chat-client.adoc" }) {
            try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
                Files.copy(in, corpus.resolve(name));
            }
        }
        corpusIngestor.ingestDirectory(corpus);
    }

    private static GoldenSet goldenSet(GoldenItem... items) {
        return new GoldenSet(Path.of("eval/golden-set.json"), "0".repeat(64), List.of(items));
    }

    private static GoldenItem item(String id, ExpectedSource expected) {
        return new GoldenItem(id, HNSW_QUESTION, List.of(expected), null, null);
    }

    @Test
    void scoresEveryQuestionAndSummarizesTheConfig() {
        GoldenItem found = item("q01", new ExpectedSource("pgvector.adoc", ""));
        GoldenItem missed = item("q02", new ExpectedSource("chat-client.adoc", "No such section"));

        EvalReport report = runner.run(goldenSet(found, missed), EvalConfig.all(properties));

        assertThat(report.run().goldenItems()).isEqualTo(2);
        assertThat(report.run().index().corpusDocuments()).isEqualTo(2);
        assertThat(report.configs()).singleElement().satisfies(config -> {
            assertThat(config.name()).isEqualTo("vector");
            assertThat(config.items()).extracting(ItemResult::id).containsExactly("q01", "q02");
            assertThat(config.items().get(0).score().firstRelevantRank()).isEqualTo(1);
            assertThat(config.items().get(1).score().firstRelevantRank()).isNull();
            assertThat(config.items()).allSatisfy(result -> assertThat(result.latencyMillis()).isNotNegative());
            assertThat(config.summary().hitAt5()).isEqualTo(0.5);
            assertThat(config.summary().mrrAt10()).isEqualTo(0.5);
        });
    }

    @Test
    void retrievesTheTopTenForMrrAt10() {
        int indexed = jdbc.sql("SELECT count(*)::int FROM vector_store").query(Integer.class).single();

        EvalReport report = runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties));

        assertThat(report.configs().getFirst().items().getFirst().retrieved())
                .hasSize(Math.min(RetrievalMetrics.MRR_K, indexed));
    }

    @Test
    void expectedSourcesThatAreNotIndexedFailTheRunInsteadOfScoringZero() {
        GoldenSet typo = goldenSet(item("q01", new ExpectedSource("api/vectordbs/pgvectr.adoc", "")));

        assertThat(runner.unindexedSources(typo)).containsExactly("api/vectordbs/pgvectr.adoc");
        assertThatThrownBy(() -> runner.run(typo, EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("api/vectordbs/pgvectr.adoc");
    }

    @Test
    void anEmptyIndexFailsTheRun() {
        jdbc.sql("TRUNCATE source_document, vector_store").update();

        assertThatThrownBy(() -> runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("POST /api/ingest/corpus");
    }
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest=ReportWriterTest`
Expected: compilation FAILURE, `cannot find symbol: class EvalReport`.

- [ ] **Step 3: Implement the runner and the report writer**

`src/main/java/com/learnings/rag/eval/EvalConfig.java`:

```java
package com.learnings.rag.eval;

import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalOptions;

/** A named retrieval configuration compared in an eval run. */
public record EvalConfig(String name, RetrievalOptions options) {

    /** The configurations every run compares. M3 has one; M4–M6 add keyword, hybrid, multi-query and rerank. */
    public static List<EvalConfig> all(RagProperties properties) {
        return List.of(new EvalConfig("vector", RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K)));
    }
}
```

`src/main/java/com/learnings/rag/eval/EvalReport.java`:

```java
package com.learnings.rag.eval;

import java.time.Instant;
import java.util.List;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.retrieval.RetrievalOptions;

/** Everything one eval run measured; written as Markdown and JSON by {@link ReportWriter}. */
public record EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs) {

    /** What was measured against, so two reports can be compared honestly. */
    public record RunInfo(String goldenSet, String goldenSetSha256, int goldenItems, String embeddingModel,
            RagProperties.Chunking chunking, IndexStats index) {
    }

    public record ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
            List<ItemResult> items) {
    }

    /** @param retrieved the ranked chunks, best first (up to {@link RetrievalMetrics#MRR_K}) */
    public record ItemResult(String id, String question, List<ExpectedSource> expectedSources,
            RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved) {
    }
}
```

`src/main/java/com/learnings/rag/eval/EvalRunner.java`:

```java
package com.learnings.rag.eval;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.RunInfo;
import com.learnings.rag.eval.RetrievalMetrics.RankedSource;
import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.ingest.SourceDocumentRepository;
import com.learnings.rag.retrieval.RetrievalPipeline;
import com.learnings.rag.retrieval.RetrievalResult;

/** Runs every golden question through the retrieval pipeline, once per configuration, and scores the results. */
@Service
public class EvalRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalRunner.class);

    private final RetrievalPipeline pipeline;
    private final ChunkCatalog catalog;
    private final SourceDocumentRepository documents;
    private final RagProperties properties;

    public EvalRunner(RetrievalPipeline pipeline, ChunkCatalog catalog, SourceDocumentRepository documents,
            RagProperties properties) {
        this.pipeline = pipeline;
        this.catalog = catalog;
        this.documents = documents;
        this.properties = properties;
    }

    public EvalReport run(GoldenSet goldenSet, List<EvalConfig> configs) {
        IndexStats index = catalog.stats();
        if (index.chunks() == 0) {
            throw new IllegalStateException("The index has no chunks for embedding model "
                    + properties.embeddingModel() + ". Run scripts/fetch-corpus.sh and POST /api/ingest/corpus first.");
        }
        List<String> unindexed = unindexedSources(goldenSet);
        if (!unindexed.isEmpty()) {
            throw new IllegalStateException("Expected sources that are not in the index: " + unindexed
                    + ". Fix the paths in " + goldenSet.path() + ", or ingest the corpus.");
        }

        Instant startedAt = Instant.now();
        List<ConfigResult> results = configs.stream().map(config -> run(goldenSet.items(), config)).toList();
        RunInfo info = new RunInfo(goldenSet.path().toString(), goldenSet.sha256(), goldenSet.items().size(),
                properties.embeddingModel(), properties.chunking(), index);
        return new EvalReport(startedAt, info, results);
    }

    /** Expected source paths missing from the index: a renamed page or a typo would otherwise silently score 0. */
    public List<String> unindexedSources(GoldenSet goldenSet) {
        return goldenSet.items().stream()
                .flatMap(item -> item.expectedSources().stream())
                .map(ExpectedSource::sourcePath)
                .distinct()
                .filter(path -> documents.findBySourcePath(path).isEmpty())
                .sorted()
                .toList();
    }

    private ConfigResult run(List<GoldenItem> items, EvalConfig config) {
        pipeline.retrieve(items.getFirst().question(), config.options()); // warm-up: pool, HTTP client, JIT
        List<ItemResult> results = new ArrayList<>(items.size());
        for (GoldenItem item : items) {
            long start = System.nanoTime();
            RetrievalResult retrieval = pipeline.retrieve(item.question(), config.options());
            long millis = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
            List<RankedSource> ranked = retrieval.documents().stream().map(EvalRunner::ranked).toList();
            results.add(new ItemResult(item.id(), item.question(), item.expectedSources(),
                    RetrievalMetrics.score(item.expectedSources(), ranked), millis, ranked));
        }
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(
                results.stream().map(ItemResult::score).toList(),
                results.stream().map(ItemResult::latencyMillis).toList());
        log.info("{}: hit@5 {} · recall@5 {} · MRR@10 {} · p50 {} ms · p95 {} ms", config.name(),
                summary.hitAt5(), summary.recallAt5(), summary.mrrAt10(), summary.p50Millis(), summary.p95Millis());
        return new ConfigResult(config.name(), config.options(), summary, results);
    }

    private static RankedSource ranked(Document document) {
        return new RankedSource(
                Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore());
    }
}
```

`src/main/java/com/learnings/rag/eval/ReportWriter.java`:

```java
package com.learnings.rag.eval;

import static java.util.stream.Collectors.joining;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.Locale;

import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.EvalReport.ConfigResult;
import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.RunInfo;

import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

/** Writes an eval report as Markdown (for reading and diffing) and JSON (every number, for later analysis). */
@Component
public class ReportWriter {

    private static final DateTimeFormatter FILE_NAME =
            DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH-mm-ss'Z'").withZone(ZoneOffset.UTC);

    private final JsonMapper json;

    public ReportWriter(JsonMapper json) {
        this.json = json;
    }

    /** Writes {@code <start time>.md} and {@code <start time>.json} into {@code directory}; returns the Markdown path. */
    public Path write(EvalReport report, Path directory) {
        String name = FILE_NAME.format(report.startedAt());
        try {
            Files.createDirectories(directory);
            Path markdown = directory.resolve(name + ".md");
            Files.writeString(markdown, markdown(report));
            Files.write(directory.resolve(name + ".json"),
                    json.writer().with(SerializationFeature.INDENT_OUTPUT).writeValueAsBytes(report));
            return markdown;
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    static String markdown(EvalReport report) {
        RunInfo run = report.run();
        RagProperties.Chunking chunking = run.chunking();
        StringBuilder md = new StringBuilder()
                .append("# Retrieval eval: ").append(report.startedAt()).append("\n\n")
                .append("| | |\n|---|---|\n")
                .append("| Golden set | `").append(run.goldenSet()).append("` (").append(run.goldenItems())
                .append(" questions, sha256 `").append(run.goldenSetSha256(), 0, 12).append("…`) |\n")
                .append("| Index | ").append(run.index().corpusDocuments()).append(" corpus pages, ")
                .append(run.index().uploadedDocuments()).append(" uploads, ").append(run.index().chunks())
                .append(" chunks |\n")
                .append("| Embedding model | `").append(run.embeddingModel()).append("` |\n")
                .append("| Chunking | max ").append(chunking.maxTokens()).append(" · min ").append(chunking.minTokens())
                .append(" · overlap ").append(chunking.overlapTokens()).append(" tokens |\n\n");
        if (run.index().uploadedDocuments() > 0) {
            md.append("> **Warning:** the index contains ").append(run.index().uploadedDocuments())
                    .append(" uploaded document(s). They compete with corpus pages for every question, so these ")
                    .append("numbers are not comparable with a corpus-only run.\n\n");
        }

        md.append("## Summary\n\n| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |\n|---|---|---|---|---|---|\n");
        for (ConfigResult config : report.configs()) {
            RetrievalMetrics.Summary summary = config.summary();
            md.append("| ").append(config.name())
                    .append(" | ").append(decimal(summary.hitAt5()))
                    .append(" | ").append(decimal(summary.recallAt5()))
                    .append(" | ").append(decimal(summary.mrrAt10()))
                    .append(" | ").append(summary.p50Millis())
                    .append(" | ").append(summary.p95Millis()).append(" |\n");
        }

        for (ConfigResult config : report.configs()) {
            md.append("\n## ").append(config.name()).append(": per question\n\n")
                    .append("| Id | First relevant rank | hit@5 | recall@5 | Question |\n|---|---|---|---|---|\n");
            for (ItemResult item : config.items()) {
                Integer rank = item.score().firstRelevantRank();
                md.append("| ").append(item.id())
                        .append(" | ").append(rank == null ? "–" : rank)
                        .append(" | ").append(item.score().hitAt5() ? "✓" : "✗")
                        .append(" | ").append(decimal(item.score().recallAt5()))
                        .append(" | ").append(cell(item.question())).append(" |\n");
            }
            List<ItemResult> misses = config.items().stream()
                    .filter(item -> item.score().firstRelevantRank() == null)
                    .toList();
            md.append("\n### Misses: no relevant chunk in the top ").append(RetrievalMetrics.MRR_K).append("\n\n");
            if (misses.isEmpty()) {
                md.append("None.\n");
            }
            for (ItemResult miss : misses) {
                md.append("- **").append(miss.id()).append("** ").append(oneLine(miss.question())).append('\n')
                        .append("  - expected: ").append(miss.expectedSources().stream()
                                .map(source -> label(source.sourcePath(), source.sectionPrefix()))
                                .collect(joining("; ")))
                        .append('\n')
                        .append("  - retrieved: ").append(miss.retrieved().stream().limit(3)
                                .map(chunk -> label(chunk.sourcePath(), chunk.breadcrumb()))
                                .collect(joining("; ")))
                        .append('\n');
            }
        }
        return md.toString();
    }

    private static String decimal(double value) {
        return String.format(Locale.ROOT, "%.3f", value);
    }

    private static String oneLine(String text) {
        return text.replaceAll("\\s+", " ").strip();
    }

    private static String cell(String text) {
        return oneLine(text).replace("|", "\\|");
    }

    private static String label(String sourcePath, String section) {
        return "`" + sourcePath + "`" + (section.isEmpty() ? "" : " › " + section);
    }
}
```

- [ ] **Step 4: Run them and watch them pass**

Run: `./mvnw -q verify -Dtest=ReportWriterTest -Dit.test=EvalRunnerIT`
Expected: PASS (6 unit tests, 4 ITs).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval
git commit -m "feat: eval runner and Markdown/JSON reports with misses and run provenance"
```

---

### Task 8: Golden and eval commands, docs, and the draft for review (M3)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/GoldenSetCommand.java`, `src/main/java/com/learnings/rag/eval/EvalCommand.java`
- Create: `src/main/resources/application-golden.yml`, `src/main/resources/application-eval.yml`, `eval/README.md`
- Modify: `README.md` (Status row for M3, a new "Evaluation" section, and `rag.eval.*` rows in Configuration)

**Interfaces:**
- Consumes: `ChunkCatalog`, `GoldenSetGenerator`, `GoldenSetFile`, `EvalRunner`, `EvalConfig`, `ReportWriter`, `EvalProperties`, `RagProperties`.
- Produces:
  - `@Profile("golden") GoldenSetCommand implements ApplicationRunner`: writes the draft and exits;
  - `@Profile("eval") EvalCommand implements ApplicationRunner`: writes a report and exits.

The commands are thin: every decision they make is already tested in Tasks 2–7. They are verified by running them.

- [ ] **Step 1: Write the commands and profiles**

`src/main/java/com/learnings/rag/eval/GoldenSetCommand.java`:

```java
package com.learnings.rag.eval;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/** {@code ./mvnw spring-boot:run -Dspring-boot.run.profiles=golden}: drafts a golden set for human review. */
@Component
@Profile("golden")
public class GoldenSetCommand implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(GoldenSetCommand.class);

    private final ChunkCatalog catalog;
    private final GoldenSetGenerator generator;
    private final GoldenSetFile files;
    private final EvalProperties properties;
    private final RagProperties ragProperties;

    public GoldenSetCommand(ChunkCatalog catalog, GoldenSetGenerator generator, GoldenSetFile files,
            EvalProperties properties, RagProperties ragProperties) {
        this.catalog = catalog;
        this.generator = generator;
        this.files = files;
        this.properties = properties;
        this.ragProperties = ragProperties;
    }

    @Override
    public void run(ApplicationArguments args) {
        EvalProperties.Golden golden = properties.golden();
        // Before any LLM call: never spend ten minutes generating a draft that would then overwrite review edits.
        files.ensureDraftWritable(golden.draft(), golden.overwrite());

        List<CorpusChunk> chunks = catalog.corpusChunks();
        if (chunks.isEmpty()) {
            throw new IllegalStateException("No corpus chunks for embedding model " + ragProperties.embeddingModel()
                    + ". Run scripts/fetch-corpus.sh and POST /api/ingest/corpus first.");
        }
        log.info("Generating {} questions from {} corpus chunks (seed {}). This calls the chat model once or "
                + "twice per question.", golden.size(), chunks.size(), golden.seed());

        List<GoldenItem> items = generator.generate(chunks);
        if (items.isEmpty()) {
            throw new IllegalStateException("No usable questions were generated; nothing was written.");
        }
        files.writeDraft(golden.draft(), items, golden.overwrite());
        log.info("Wrote {} questions to {}. Review them as described in eval/README.md, then save the result as {}.",
                items.size(), golden.draft(), properties.goldenSet());
    }
}
```

`src/main/java/com/learnings/rag/eval/EvalCommand.java`:

```java
package com.learnings.rag.eval;

import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/** {@code ./mvnw spring-boot:run -Dspring-boot.run.profiles=eval}: scores retrieval against the golden set. */
@Component
@Profile("eval")
public class EvalCommand implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalCommand.class);

    private final GoldenSetFile files;
    private final EvalRunner runner;
    private final ReportWriter reportWriter;
    private final EvalProperties properties;
    private final RagProperties ragProperties;

    public EvalCommand(GoldenSetFile files, EvalRunner runner, ReportWriter reportWriter, EvalProperties properties,
            RagProperties ragProperties) {
        this.files = files;
        this.runner = runner;
        this.reportWriter = reportWriter;
        this.properties = properties;
        this.ragProperties = ragProperties;
    }

    @Override
    public void run(ApplicationArguments args) {
        GoldenSet goldenSet = files.read(properties.goldenSet());
        log.info("Evaluating {} questions from {}", goldenSet.items().size(), goldenSet.path());
        EvalReport report = runner.run(goldenSet, EvalConfig.all(ragProperties)); // logs one summary line per config
        Path markdown = reportWriter.write(report, properties.reportsDir());
        log.info("Report written to {}", markdown);
    }
}
```

`src/main/resources/application-golden.yml`:

```yaml
# ./mvnw spring-boot:run -Dspring-boot.run.profiles=golden
# Drafts eval/golden-set.draft.json from the indexed corpus, then exits. No web server.
spring:
  main:
    web-application-type: none
```

`src/main/resources/application-eval.yml`:

```yaml
# ./mvnw spring-boot:run -Dspring-boot.run.profiles=eval
# Scores retrieval against eval/golden-set.json, writes eval/reports/<time>.md + .json, then exits. No web server.
spring:
  main:
    web-application-type: none
```

- [ ] **Step 2: Write `eval/README.md`**

````markdown
# Evaluation

`golden-set.json` is the hand-reviewed question set that every retrieval change is measured against. Each question
names the section(s) that answer it (`sourcePath` + heading-path `sectionPrefix`), not chunk ids, so the set stays
valid when chunking changes.

## 1. Generate a draft

The corpus must be indexed first (`scripts/fetch-corpus.sh`, then `POST /api/ingest/corpus`).

```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=golden
```

This writes `eval/golden-set.draft.json`, which is gitignored and holds 40 questions:
- One chunk is sampled per page, in a seeded random order, so a long page gets no more questions than a short one.
- The chat model writes a developer-style question and a short reference answer for each chunk.
- A question that copies five or more consecutive words from its chunk is rewritten once, then dropped. Identifiers
  count as one word. Copied wording would make keyword search look better than it is.

It takes a few minutes and costs a few cents. To use a stronger model for writing the questions, run
`OPENAI_CHAT_MODEL=gpt-5 ./mvnw …`. The command refuses to overwrite an existing draft; add
`-Dspring-boot.run.arguments=--rag.eval.golden.overwrite=true` to replace it.

## 2. Review the draft and save it as `golden-set.json`

For every item:

- **Keep it** only if a developer would plausibly ask it and the `sourceExcerpt` really answers it. Delete weak, vague
  or trivia questions. Aim to keep at least 30.
- **Fix the wording** so it sounds like a real search. Don't paste phrases from the excerpt. Naming a class, property or
  annotation is fine.
- **Check `expectedSources`.** List every section that answers the question:
  - `sectionPrefix` is a heading path. A chunk matches when its breadcrumb equals the prefix or sits under it, so
    `Indexes` matches `Indexes › HNSW` but not `Index`.
  - Shorten the prefix to a parent heading when the answer spans its subsections. `""` accepts any chunk of the page.
  - If another page also answers the question, add it. hit@5 then needs either source; recall@5 counts both.
- **`referenceAnswer`** may be edited. M7's answer evals will use it.
- **`sourceExcerpt`** is context for you and is never scored.
- **Optionally add your own items.** Good candidates are identifier lookups (*"what does
  spring.ai.vectorstore.pgvector.index-type do?"*), troubleshooting questions, and questions whose answer spans two
  pages. Give each one a unique `id` and look up its sources.

Save the result as `eval/golden-set.json` and commit it. The eval rejects the file, listing the item ids, if an item has:
- a missing or duplicate `id`;
- an empty question;
- no `expectedSources`;
- a source without `sectionPrefix` (for example because of a typo in the field name);
- a `sourcePath` that isn't indexed.

## 3. Run the eval

```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=eval
```

This writes `eval/reports/<UTC time>.md` and `.json` (gitignored). Each report records:
- the golden set's sha256;
- the index composition, with a warning if uploads are mixed in;
- the embedding model and the chunking settings;
- per-question ranks and a list of misses (what was expected and what came back).

## Metrics

| Metric | Meaning |
|---|---|
| hit@5 | Share of questions with at least one relevant chunk in the top 5, which is what chat hands the model. |
| recall@5 | Per question, the share of its expected sources found in the top 5, averaged. |
| MRR@10 | Mean of 1 / rank of the first relevant chunk within the top 10 (0 if none). |
| p50 / p95 | Retrieval latency per question in ms, including the query-embedding call (nearest-rank percentiles). |

## Baseline

Filled in when the first report is recorded (M3).
````

- [ ] **Step 3: Update `README.md`**

Make these four edits:
1. In the **Status** table, change the M3 row's state from `next` to `in progress`.
2. Add `- [Evaluation](#evaluation)` to **Contents**, after `- [Testing](#testing)`.
3. Insert this section after the **Testing** section:

```markdown
## Evaluation

Retrieval quality is measured against a hand-reviewed golden set ([`eval/README.md`](eval/README.md)):

```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=golden   # draft eval/golden-set.draft.json (LLM-written questions)
# review the draft → save as eval/golden-set.json
./mvnw spring-boot:run -Dspring-boot.run.profiles=eval     # report: eval/reports/<time>.md + .json
```

The eval reports hit@5, recall@5, MRR@10 and p50/p95 retrieval latency for each retrieval configuration. Golden
labels name a page and a heading path, not chunk ids, so the set survives re-chunking.
```

4. Append these rows to the **Configuration** table:

```markdown
| `rag.eval.golden-set` | `eval/golden-set.json` | The reviewed golden set the eval scores against. |
| `rag.eval.reports-dir` | `eval/reports` | Where eval reports are written (gitignored). |
| `rag.eval.golden.size` / `.seed` | `40` / `42` | Questions to draft, and the sampling seed. |
| `rag.eval.golden.overwrite` | `false` | Allow the generator to replace an existing draft. |
```

- [ ] **Step 4: Run the full suite**

Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS. The commands are inactive without their profiles.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/main/resources/application-golden.yml src/main/resources/application-eval.yml eval/README.md README.md
git commit -m "feat: golden and eval commands as non-web profiles; evaluation docs"
```

- [ ] **Step 6: Generate the real draft (calls OpenAI)**

Preconditions: Docker is running, the corpus is indexed (`curl -s localhost:8081/api/documents` lists 52 pages, or run `POST /api/ingest/corpus`), and no app is running on 8081. The golden profile doesn't need the web server.

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=golden`
Expected:
- a log line per accepted question, e.g. `12/40 api/vectordbs/pgvector.adoc › …: <question>`;
- then `Wrote 40 questions to eval/golden-set.draft.json`;
- the process exits by itself.

If the process doesn't exit, record a ruling and add an explicit `System.exit(SpringApplication.exit(context))` to both commands.

Check: `python3 -c "import json; d=json.load(open('eval/golden-set.draft.json')); print(len(d), len({i['expectedSources'][0]['sourcePath'] for i in d}))"`
Expected: `40 40`, or close to it. That is 40 items from about 40 distinct pages.

- [ ] **Step 7: PAUSE: the human reviews the draft**

Stop here. Show the user the draft's path, how many items it has, a sample of five questions, and the review checklist in `eval/README.md` §2. **Do not start Task 9 until the user has saved `eval/golden-set.json`.**

---

### Task 9: Record the vector-only baseline (M3)

**Files:**
- Create: `eval/golden-set.json` (the user's reviewed set; commit it as the user saved it)
- Modify: `eval/README.md` (Baseline section), `README.md` (Status: M3 done; Evaluation: baseline table), `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md` (only if the review changed a decision)

**Interfaces:**
- Consumes: `EvalCommand` (Task 8); the reviewed `eval/golden-set.json`.

- [ ] **Step 1: Validate and run the eval (calls OpenAI for 1 + N query embeddings)**

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval`
Expected:
- The log shows `Evaluating <N> questions`, then `vector: hit@5 … · recall@5 … · MRR@10 … · p50 … ms · p95 … ms`, then `Report written to eval/reports/<time>.md`.
- The process exits.
- If it fails with "has N problem(s)" or "Expected sources that are not in the index", show the user the message and the item ids. They are review mistakes for the user to fix; do not edit their golden set on their behalf.

- [ ] **Step 2: Read the report**

Open the Markdown report. Check that:
- the header shows the reviewed `golden-set.json` with N questions, 52 corpus pages, `0 uploads` and 1,106 chunks (or the current numbers);
- the summary has one `vector` row;
- the **Misses** section lists questions with nothing relevant in the top 10.

For each miss, decide whether retrieval failed (keep it: that's what M4–M6 are for) or the label is wrong (raise it with the user). Do not tune labels to improve the score.

- [ ] **Step 3: Record the baseline**

In `eval/README.md`, replace `Filled in when the first report is recorded (M3).` with the report's header table and summary table, copied verbatim, followed by this line:

`Baseline: vector-only retrieval (M2 pipeline), top 10, text-embedding-3-small. Every later configuration is compared with this row on the same golden set (sha256 above).`

In `README.md`:
- change the M3 Status row to `✅ done`;
- under **Evaluation**, add `**Baseline (vector only):**` followed by the summary table copied from the report.

- [ ] **Step 4: Run the full suite**

Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/golden-set.json eval/README.md README.md
git commit -m "eval: reviewed golden set and the vector-only baseline (M3)"
```

---

## After this plan

M3 is complete when Task 9 Step 3 records the baseline. Next is **M4, hybrid retrieval**: a `KeywordRetriever` (`websearch_to_tsquery` + `ts_rank_cd` over the existing `content_tsv` column) and `ReciprocalRankFusion`, with `rag.retrieval.mode=VECTOR|KEYWORD|HYBRID`. M4 adds `keyword` and `hybrid` to `EvalConfig.all`, so its first report compares all three against this baseline. It gets its own plan.
