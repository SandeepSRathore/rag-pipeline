# RAG Pipeline M4: Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add keyword search and reciprocal rank fusion (`rag.retrieval.mode=VECTOR|KEYWORD|HYBRID`), extend the golden set with identifier questions, and produce the report the spec asks for: hybrid vs vector vs keyword. The default chat mode follows the result.

**Architecture:**
- **`KeywordRetriever`** queries the existing generated `content_tsv` column with JdbcClient. It uses `websearch_to_tsquery` parsing with OR semantics, ranks with `ts_rank_cd`, and is filtered to the current embedding model.
- **`ReciprocalRankFusion`** is a pure function: RRF with k = 60.
- **`RetrievalPipeline`** switches on `RetrievalOptions.mode`. HYBRID runs both retrievers in parallel on virtual threads (20 candidates each) and fuses them into the top K. The trace records the stages `vector`, `keyword` and `fusion` plus a wall-clock total.
- **Chat sources** carry each stage's score, so the UI shows "similarity · keyword · fusion" instead of a single, now ambiguous, number.
- **The golden set** gains optional `tags` and 12 identifier questions. Reports gain a per-tag table.

**Tech Stack:** Java 25, Spring Boot 4.1.1, Spring AI 2.0.1, PostgreSQL 17 full-text search (`websearch_to_tsquery`, `ts_rank_cd`, GIN index from V1), JdbcClient, Jackson 3, virtual threads, Testcontainers, JUnit 5, AssertJ, Mockito.

**Spec:** `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`. This plan covers milestone M4 and the Architecture steps ③ (retrieve) and ④ (join) for a single query; multi-query is M5. It adds amendments 22–26.

## Global Constraints

- Java 25; Spring Boot `4.1.1`; `spring-ai-bom` `2.0.1`; `./mvnw`.
- Root package `com.learnings.rag`; `@ConfigurationProperties` records under `rag.*`.
- Jackson 3 only (`tools.jackson.*`).
- Main code refers to chunk metadata keys only through `ChunkMetadata` constants.
- `*Test` = unit tests without Docker; `*IT` = Testcontainers ITs. Tests never call OpenAI.
- Spec values: RRF `k = 60`; 20 candidates per retriever before fusion; keyword search uses `websearch_to_tsquery` + `ts_rank_cd`.
- Both retrievers search only chunks of the current embedding model (`ChunkMetadata.EMBEDDING_MODEL`).
- Learning project: no auth or rate limits. App on `127.0.0.1:8081`. Work on branch `m4-hybrid`.

**Spec amendments made while planning (also recorded in the spec):**
22. **Keyword search uses OR semantics.**
    - The question is parsed by `websearch_to_tsquery('english', …)`, which handles stop words, stemming, quoted phrases and `-negation`. The parsed `&`s are then replaced with `|`, and `ts_rank_cd` ranks the results.
    - Probe on the real index: with AND, 2 of 4 natural-language questions matched **0** chunks; with OR they matched hundreds, ranked.
    - Dotted identifiers stay single lexemes (`'spring.ai.vectorstore.pgvector.index'`), so exact-identifier questions still match precisely.
23. **HYBRID runs vector and keyword search in parallel** on virtual threads, `candidates` (20) each, and fuses them with RRF (k = 60) into `topK`.
    - Ties keep first-seen order, which puts vector first.
    - `PipelineTrace.totalMillis` is wall-clock time, not the sum of the stages.
24. **`rag.retrieval.mode` stays `VECTOR` until the M4 report decides.**
    - HYBRID becomes the default if its MRR@10 ≥ vector's and its hit@5 is no more than one question below vector's (one question = 1/N, the measurement's noise).
    - Otherwise VECTOR stays, and the report says why.
25. **Golden items take optional `tags`.**
    - 12 identifier questions (`i01`–`i12`) and `h01` are tagged `identifier`.
    - Reports add a per-tag table (`identifier`, `untagged`).
    - M4 compares configurations on this 53-item set. The M3 baseline (41 items) stays recorded as history.
26. **Chat sources carry `scores`**, a map from stage to score (`vector`, `keyword`, `fusion`). The UI shows them, labelling the vector score "similarity".

## Review Focus

Inputs the spec implies but doesn't spell out, most likely to bite first. Each one gets a test in the task that owns it:

1. **A question made only of stop words or punctuation** (`how do I do it?`) produces an empty `tsquery`. Keyword search must return nothing rather than fail, and HYBRID must still answer from vector search. Covered by `stopWordsOnlyReturnNothing` (Task 3) and `hybridStillAnswersWhenKeywordSearchFindsNothing` (Task 5).
2. **Questions containing tsquery syntax** (`'`, `&`, `|`, `!`, `:*`, parentheses, unbalanced quotes) must be treated as text, not crash the query. Covered by `queryOperatorsInTheQuestionAreTreatedAsText` (Task 3).
3. **Keyword search must not return chunks of another embedding model.** Otherwise HYBRID would mix stale chunks back in after a model switch. Covered by `onlySearchesChunksOfTheCurrentEmbeddingModel` (Task 3).
4. **A failure in one of the parallel retrievers** (query embedding unavailable, DB error) must surface as the original exception: not hang, not be swallowed, not be wrapped beyond recognition. Covered by `aFailingRetrieverFailsTheHybridSearchWithItsOwnException` (Task 5).
5. **A chunk returned by both retrievers** must appear once in the fused list, with both contributions summed. Covered by `aDocumentInBothRankingsAppearsOnceWithSummedScore` (Task 4).

---

## File map

```
src/main/java/com/learnings/rag/
  config/RagProperties.java                    Retrieval gains mode + candidates (modify)
  retrieval/RetrievalMode.java                 VECTOR | KEYWORD | HYBRID (new)
  retrieval/RetrievalOptions.java              + mode, candidates, withMode (modify)
  retrieval/PipelineTrace.java                 wall-clock totalMillis component (modify)
  retrieval/KeywordRetriever.java              Postgres FTS, OR semantics, ts_rank_cd (new)
  retrieval/ReciprocalRankFusion.java          RRF k=60 (new)
  retrieval/RetrievalPipeline.java             mode switch, parallel hybrid (modify)
  generation/SourceRef.java, ChatEvent.java, AnswerService.java   per-stage scores (modify)
  eval/GoldenItem.java, GoldenSetFile.java     tags (modify)
  eval/EvalReport.java, EvalRunner.java, ReportWriter.java        per-tag summaries (modify)
  eval/EvalConfig.java                         vector, keyword, hybrid (modify)
src/main/resources/application.yml             rag.retrieval.mode / candidates
src/main/resources/static/app.js               per-stage scores in the sources panel
src/test/java/com/learnings/rag/
  retrieval/{KeywordRetrieverIT, ReciprocalRankFusionTest, RetrievalPipelineTest}.java (new)
  retrieval/{RetrievalPipelineIT, RetrievalOptionsTest}.java (modify)
  eval/{EvalRunnerTagsTest}.java (new); eval/{GoldenSetFileTest, ReportWriterTest, EvalRunnerIT}.java (modify)
  generation/AnswerServiceTest.java (modify)
eval/golden-set.json, eval/README.md, README.md, spec
```

---

### Task 1: Tagged golden items and per-tag summaries (M4)

**Files:**
- Modify: `src/main/java/com/learnings/rag/eval/GoldenItem.java`, `GoldenSetFile.java`, `EvalReport.java`, `EvalRunner.java`, `ReportWriter.java`
- Test: `src/test/java/com/learnings/rag/eval/EvalRunnerTagsTest.java` (new), `GoldenSetFileTest.java`, `ReportWriterTest.java`

**Interfaces:**
- Produces:
  - `GoldenItem`:
    - gains a final component `List<String> tags`; null becomes `List.of()`;
    - the 5-argument constructor stays and means untagged.
  - `EvalReport`:
    - `record EvalReport.TagSummary(String tag, RetrievalMetrics.Summary summary)`;
    - `EvalReport.ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary, List<TagSummary> byTag, List<ItemResult> items)`.
  - `EvalRunner`:
    - `static List<TagSummary> summarizeByTag(List<GoldenItem>, List<ItemResult>)`, which is empty when no item is tagged;
    - `static final String UNTAGGED = "untagged"`.

- [ ] **Step 1: Write the failing tests**

`src/test/java/com/learnings/rag/eval/EvalRunnerTagsTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.learnings.rag.eval.EvalReport.ItemResult;
import com.learnings.rag.eval.EvalReport.TagSummary;
import com.learnings.rag.eval.RetrievalMetrics.ItemScore;

class EvalRunnerTagsTest {

    private static final ExpectedSource SOURCE = new ExpectedSource("a.adoc", "");

    private static GoldenItem item(String id, List<String> tags) {
        return new GoldenItem(id, "Question " + id + "?", List.of(SOURCE), null, null, tags);
    }

    private static ItemResult result(String id, boolean hit) {
        ItemScore score = hit ? new ItemScore(1, true, 1.0, 1.0) : new ItemScore(null, false, 0.0, 0.0);
        return new ItemResult(id, "Question " + id + "?", List.of(SOURCE), score, 100, List.of());
    }

    @Test
    void summarizesEachTagAndTheUntaggedRest() {
        List<GoldenItem> items = List.of(item("q1", List.of()), item("i1", List.of("identifier")),
                item("i2", List.of("identifier")), item("q2", null));
        List<ItemResult> results = List.of(result("q1", true), result("i1", true), result("i2", false),
                result("q2", true));

        List<TagSummary> byTag = EvalRunner.summarizeByTag(items, results);

        assertThat(byTag).extracting(TagSummary::tag).containsExactly("identifier", EvalRunner.UNTAGGED);
        assertThat(byTag.get(0).summary().items()).isEqualTo(2);
        assertThat(byTag.get(0).summary().hitAt5()).isEqualTo(0.5);
        assertThat(byTag.get(1).summary().items()).isEqualTo(2);
        assertThat(byTag.get(1).summary().hitAt5()).isEqualTo(1.0);
    }

    @Test
    void anItemWithSeveralTagsCountsUnderEach() {
        List<TagSummary> byTag = EvalRunner.summarizeByTag(List.of(item("i1", List.of("identifier", "table"))),
                List.of(result("i1", true)));

        assertThat(byTag).extracting(TagSummary::tag).containsExactly("identifier", "table");
    }

    @Test
    void noTaggedItemsMeansNoPerTagSummaries() {
        assertThat(EvalRunner.summarizeByTag(List.of(item("q1", List.of())), List.of(result("q1", true)))).isEmpty();
    }
}
```

In `src/test/java/com/learnings/rag/eval/GoldenSetFileTest.java`, add these tests after `anEmptySetIsInvalid`:

```java
    @Test
    void tagsAreOptionalAndRoundTrip() throws IOException {
        Path path = dir.resolve("golden-set.json");
        Files.writeString(path, """
                [
                  {"id": "q01", "question": "How do I enable HNSW?",
                   "expectedSources": [{"sourcePath": "a.adoc", "sectionPrefix": ""}]},
                  {"id": "i01", "question": "What does spring.ai.x do?",
                   "expectedSources": [{"sourcePath": "a.adoc", "sectionPrefix": ""}], "tags": ["identifier"]}
                ]""");

        GoldenSet set = file.read(path);

        assertThat(set.items()).extracting(GoldenItem::tags).containsExactly(List.of(), List.of("identifier"));
    }

    @Test
    void blankTagsAreRejected() throws IOException {
        Path path = dir.resolve("golden-set.json");
        Files.writeString(path, """
                [{"id": "q01", "question": "How?", "expectedSources": [{"sourcePath": "a.adoc", "sectionPrefix": ""}],
                  "tags": [" "]}]""");

        assertThatThrownBy(() -> file.read(path)).hasMessageContaining("q01: blank tag");
    }
```

In `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`, make two changes.

First, in `report(int uploads)`, replace the `return new EvalReport(...)` statement with:

```java
        List<EvalReport.TagSummary> byTag = List.of(
                new EvalReport.TagSummary("identifier", RetrievalMetrics.summarize(List.of(miss.score()), List.of(80L))),
                new EvalReport.TagSummary("untagged", RetrievalMetrics.summarize(List.of(hit.score()), List.of(120L))));
        return new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), run,
                List.of(new ConfigResult("vector", new RetrievalOptions(10, 0.0), summary, byTag, List.of(hit, miss))));
```

Second, add this test:

```java
    @Test
    void byTagTableShowsEachTagPerConfig() {
        assertThat(ReportWriter.markdown(report(0))).contains(
                "## By tag",
                "| vector | identifier | 1 | 0.000 | 0.000 | 0.000 |",
                "| vector | untagged | 1 | 1.000 | 1.000 | 1.000 |");
    }
```

`RetrievalOptions` gains `mode` and `candidates` in Task 5, whose Step 1 updates this call.

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest='EvalRunnerTagsTest,GoldenSetFileTest,ReportWriterTest'`
Expected: compilation FAILURE (`GoldenItem` has no 6-argument constructor; `cannot find symbol: class TagSummary`).

- [ ] **Step 3: Implement tags and per-tag summaries**

Replace `src/main/java/com/learnings/rag/eval/GoldenItem.java` with:

```java
package com.learnings.rag.eval;

import java.util.List;

/**
 * One golden question.
 *
 * @param expectedSources every section that answers the question (relevant locations, as in IR): hit@5 needs any of
 *        them in the top 5; recall@5 is the share of them found there
 * @param referenceAnswer a short answer written from the source chunk; used by the answer evals in M7
 * @param sourceExcerpt the chunk the question was generated from, as context for reviewers; never scored
 * @param tags optional labels such as {@code identifier}; reports summarize each tag separately
 */
public record GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
        String sourceExcerpt, List<String> tags) {

    /** A missing tags field (older files, generated drafts) means untagged. */
    public GoldenItem {
        tags = tags == null ? List.of() : tags;
    }

    public GoldenItem(String id, String question, List<ExpectedSource> expectedSources, String referenceAnswer,
            String sourceExcerpt) {
        this(id, question, expectedSources, referenceAnswer, sourceExcerpt, List.of());
    }
}
```

In `src/main/java/com/learnings/rag/eval/GoldenSetFile.java`, inside `problems(...)`, find this line:

```java
            if (item.expectedSources() == null || item.expectedSources().isEmpty()) {
```

and insert this block directly above it:

```java
            if (item.tags().stream().anyMatch(tag -> tag == null || tag.isBlank())) {
                problems.add(label + ": blank tag");
            }
```

Replace `src/main/java/com/learnings/rag/eval/EvalReport.java` with:

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

    /** @param byTag one summary per tag (plus "untagged"); empty when no golden item is tagged */
    public record ConfigResult(String name, RetrievalOptions options, RetrievalMetrics.Summary summary,
            List<TagSummary> byTag, List<ItemResult> items) {
    }

    public record TagSummary(String tag, RetrievalMetrics.Summary summary) {
    }

    /** @param retrieved the ranked chunks, best first (up to {@link RetrievalMetrics#MRR_K}) */
    public record ItemResult(String id, String question, List<ExpectedSource> expectedSources,
            RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved) {
    }
}
```

In `src/main/java/com/learnings/rag/eval/EvalRunner.java`, make these edits:

1. Add the imports `java.util.TreeMap` and `com.learnings.rag.eval.EvalReport.TagSummary`.
2. Add the constant `static final String UNTAGGED = "untagged";` below the `log` field.
3. Change the `run(List<GoldenItem> items, EvalConfig config)` method's last statement from `return new ConfigResult(config.name(), config.options(), summary, results);` to:

```java
        return new ConfigResult(config.name(), config.options(), summary, summarizeByTag(items, results), results);
```

4. Add this method below `run(List<GoldenItem>, EvalConfig)`:

```java
    /** One summary per tag, plus "untagged" for the rest, in tag order; empty when no item is tagged. */
    static List<TagSummary> summarizeByTag(List<GoldenItem> items, List<ItemResult> results) {
        if (items.stream().allMatch(item -> item.tags().isEmpty())) {
            return List.of();
        }
        Map<String, List<ItemResult>> groups = new TreeMap<>();
        for (int i = 0; i < items.size(); i++) {
            List<String> tags = items.get(i).tags().isEmpty() ? List.of(UNTAGGED) : items.get(i).tags();
            for (String tag : tags) {
                groups.computeIfAbsent(tag, key -> new ArrayList<>()).add(results.get(i));
            }
        }
        return groups.entrySet().stream()
                .map(group -> new TagSummary(group.getKey(), RetrievalMetrics.summarize(
                        group.getValue().stream().map(ItemResult::score).toList(),
                        group.getValue().stream().map(ItemResult::latencyMillis).toList())))
                .toList();
    }
```

In `src/main/java/com/learnings/rag/eval/ReportWriter.java`, add the import `com.learnings.rag.eval.EvalReport.TagSummary`. Then find the line that starts the per-question sections:

```java
        for (ConfigResult config : report.configs()) {
            md.append("\n## ").append(config.name()).append(": per question\n\n")
```

and insert this block directly above it:

```java
        if (report.configs().stream().anyMatch(config -> !config.byTag().isEmpty())) {
            md.append("\n## By tag\n\n| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |\n|---|---|---|---|---|---|\n");
            for (ConfigResult config : report.configs()) {
                for (TagSummary tag : config.byTag()) {
                    RetrievalMetrics.Summary summary = tag.summary();
                    md.append("| ").append(config.name())
                            .append(" | ").append(tag.tag())
                            .append(" | ").append(summary.items())
                            .append(" | ").append(decimal(summary.hitAt5()))
                            .append(" | ").append(decimal(summary.recallAt5()))
                            .append(" | ").append(decimal(summary.mrrAt10())).append(" |\n");
                }
            }
        }
```

- [ ] **Step 4: Run them and watch them pass**

Run: `./mvnw -q test -Dtest='EvalRunnerTagsTest,GoldenSetFileTest,ReportWriterTest'`
Expected: PASS (3 + 8 + 7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval
git commit -m "feat: optional golden-item tags and per-tag summaries in eval reports"
```

---

### Task 2: Identifier questions in the golden set (M4)

**Files:**
- Modify: `eval/golden-set.json` (12 new items, `h01` tagged), `eval/README.md`

**Interfaces:**
- Consumes: `GoldenItem.tags` (Task 1); `EvalCommand` (M3).

Every label below was checked against the index while planning: each property name appears only in the listed sections. Each question names a property the way a developer would type it into a search box.

- [ ] **Step 1: Add the identifier items**

Run this from the repo root:

```bash
python3 - <<'EOF'
import json
path = 'eval/golden-set.json'
items = json.load(open(path))
assert not any(i['id'].startswith('i') for i in items), 'identifier items already added'

def item(id, question, sources, answer):
    return {"id": id, "question": question,
            "expectedSources": [{"sourcePath": p, "sectionPrefix": s} for p, s in sources],
            "referenceAnswer": answer, "sourceExcerpt": None, "tags": ["identifier"]}

items += [
    item("i01", "What does spring.ai.chat.client.observations.log-completion do?",
         [("observability/index.adoc", "Chat Client › Prompt and Completion Data")],
         "It controls whether the chat client's completion content is logged; it is false by default."),
    item("i02", "spring.ai.vectorstore.milvus.client.rpc-deadline-ms",
         [("api/vectordbs/milvus.adoc", "Milvus VectorStore properties")],
         "How long the client waits for a reply from the Milvus server; with a deadline set, the client waits through fast RPC failures caused by network fluctuations."),
    item("i03", "What does spring.ai.chat.memory.cassandra.time-to-live control?",
         [("api/chat-memory.adoc", "Memory Storage › CassandraChatMemoryRepository")],
         "The time to live (TTL) of chat-memory messages written to Cassandra."),
    item("i04", "What is the default of spring.ai.vectorstore.neo4j.embedding-dimension?",
         [("api/vectordbs/neo4j.adoc", "Auto-configuration › Configuration Properties")],
         "1536: the number of dimensions in the stored vectors."),
    item("i05", "What can I set spring.ai.openai.chat.output-modalities to?",
         [("api/chat/openai-chat.adoc", "Auto-configuration › Chat Properties › Configuration Properties")],
         "The output types the model should generate for the request; text is the default."),
    item("i06", "spring.ai.chat.client.tool-search-advisor.max-results default",
         [("api/tools/tool-search-tool.adoc", "Spring Boot Auto-Configuration › Configuration Properties Reference")],
         "The maximum number of tool references returned per search call; null, the default, uses the built-in limit."),
    item("i07", "What is the default spring.ai.retry.backoff.max-interval?",
         [("api/chat/openai-chat.adoc", "Auto-configuration › Chat Properties › Retry Properties"),
          ("api/embeddings/openai-embeddings.adoc", "Auto-configuration › Embedding Properties › Retry Properties")],
         "3 minutes: the maximum backoff duration between retries."),
    item("i08", "What happens if I set spring.ai.tools.limits.max-calls-per-tool-default to -1?",
         [("api/tools.adoc", "Tool Specification Reference › Tool Call Limits")],
         "-1 disables the per-turn limit on calls to one tool; the default limit is 40."),
    item("i09", "spring.ai.vectorstore.redis.semantic-cache.prefix",
         [("api/vectordbs/redis.adoc", "Semantic Caching › Configuration Properties")],
         "The key prefix for cached entries in Redis; semantic-cache: by default."),
    item("i10", "Where does spring.ai.embedding.transformer.cache.directory keep ONNX models by default?",
         [("api/embeddings/onnx.adoc", "Auto-configuration › Embedding Properties")],
         "In ${java.io.tmpdir}/spring-ai-onnx-model, the directory used to cache remote resources such as ONNX models."),
    item("i11", "How do I cap web searches per request with spring.ai.anthropic.chat.web-search-tool.max-uses?",
         [("api/chat/anthropic-chat.adoc", "Auto-Configuration › Configuration Properties"),
          ("api/chat/anthropic-chat.adoc", "Web Search › Spring Boot Configuration")],
         "Set spring.ai.anthropic.chat.web-search-tool.max-uses to the maximum number of web searches allowed per request, e.g. 5."),
    item("i12", "What does spring.ai.ollama.chat.num-predict -1 mean?",
         [("api/chat/ollama-chat.adoc", "Auto-configuration › Chat Properties")],
         "num-predict is the maximum number of tokens to generate; -1 (the default) means unlimited generation and -2 fills the context."),
]
for i in items:
    if i['id'] == 'h01':
        i['tags'] = ['identifier']
json.dump(items, open(path, 'w'), indent=2, ensure_ascii=False)
open(path, 'a').write('\n')
print(len(items), 'items;', sum('identifier' in (i.get('tags') or []) for i in items), 'tagged identifier')
EOF
```

Expected: `53 items; 13 tagged identifier`.

- [ ] **Step 2: Validate the labels against the index (calls OpenAI for query embeddings)**

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval`
Expected:
- the run completes with `Report written to eval/reports/<time>.md`;
- the report has a `## By tag` table with `identifier | 13` and `untagged | 40` rows for `vector`.

If it fails with "Expected sources that match no indexed chunk", the corpus or chunker changed since planning: fix the listed labels against the current index.

- [ ] **Step 3: Document the change in `eval/README.md`**

Insert this paragraph directly above the `## Baseline` heading:

```markdown
## Golden set versions

- **v1 (M3, 41 items):** generated, AI-reviewed, plus `h01`–`h04`. The M3 baseline below was measured on v1.
- **v2 (M4, 53 items):** adds `i01`–`i12`. Each is a question naming a Spring AI property the way a developer would
  type it, and each label was checked against the index while planning. These items and `h01` carry
  `"tags": ["identifier"]`, so reports show the identifier questions separately from the rest (`untagged`). Keyword
  search is expected to help most on identifiers.
```

- [ ] **Step 4: Commit**

```bash
git add eval/golden-set.json eval/README.md
git commit -m "eval: 12 identifier questions tagged for per-tag reporting (golden set v2)"
```

---

### Task 3: Keyword retriever (M4)

**Files:**
- Create: `src/main/java/com/learnings/rag/retrieval/KeywordRetriever.java`
- Test: `src/test/java/com/learnings/rag/retrieval/KeywordRetrieverIT.java`

**Interfaces:**
- Consumes: `vector_store.content_tsv` + GIN index (V1); `ChunkMetadata.EMBEDDING_MODEL`; `RagProperties.embeddingModel()`; the Spring Boot `JsonMapper` bean.
- Produces: `@Component KeywordRetriever(JdbcClient, JsonMapper, RagProperties)` with `List<Document> retrieve(String question, int limit)`.
  - Each Document has the vector_store row's `id`, the `content` as text, the parsed `metadata`, and `score` = `ts_rank_cd`.
  - Results come best first, with ties broken by id.

- [ ] **Step 1: Write the failing integration test**

`src/test/java/com/learnings/rag/retrieval/KeywordRetrieverIT.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.ai.document.Document;
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
class KeywordRetrieverIT {

    @Autowired
    CorpusIngestor corpusIngestor;

    @Autowired
    KeywordRetriever keywordRetriever;

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
    void ingestFixtures() throws IOException {
        jdbc.sql("TRUNCATE source_document, vector_store").update();
        for (String name : new String[] { "pgvector.adoc", "chat-client.adoc" }) {
            try (InputStream in = new ClassPathResource("fixtures/corpus/" + name).getInputStream()) {
                Files.copy(in, corpus.resolve(name));
            }
        }
        corpusIngestor.ingestDirectory(corpus);
    }

    private static String sourcePath(Document document) {
        return (String) document.getMetadata().get(ChunkMetadata.SOURCE_PATH);
    }

    @Test
    void anExactIdentifierFindsTheChunkThatContainsIt() {
        List<Document> results = keywordRetriever.retrieve("spring.ai.vectorstore.pgvector.index-type", 5);

        assertThat(results).isNotEmpty();
        assertThat(results.getFirst().getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Configuration properties");
        assertThat(results.getFirst().getText()).contains("spring.ai.vectorstore.pgvector.index-type");
    }

    @Test
    void matchesChunksContainingAnyTermNotOnlyAllOfThem() {
        // "zebracorn" occurs in no chunk: with AND semantics this question would match nothing.
        List<Document> results = keywordRetriever.retrieve("How do I stream responses with a zebracorn?", 10);

        assertThat(results).anySatisfy(document -> assertThat(document.getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "chat-client.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Streaming responses"));
    }

    @Test
    void scoresAreTheRankAndResultsComeBestFirst() {
        List<Document> results = keywordRetriever.retrieve("HNSW index graph memory", 10);

        assertThat(results).isNotEmpty();
        assertThat(results).allSatisfy(document -> assertThat(document.getScore()).isPositive());
        assertThat(results).extracting(Document::getScore).isSortedAccordingTo((a, b) -> Double.compare(b, a));
    }

    @Test
    void limitCapsTheResults() {
        assertThat(keywordRetriever.retrieve("spring ai chat client vector store", 2)).hasSizeLessThanOrEqualTo(2);
    }

    @Test
    void stopWordsOnlyReturnNothing() {
        assertThat(keywordRetriever.retrieve("how do I do it?", 10)).isEmpty();
        assertThat(keywordRetriever.retrieve("?!", 10)).isEmpty();
    }

    @Test
    void queryOperatorsInTheQuestionAreTreatedAsText() {
        for (String question : List.of("HNSW & graph | !memory", "it's \"unbalanced", "index:* (type)", "a & | ! b")) {
            assertThat(keywordRetriever.retrieve(question, 5)).as(question).isNotNull();
        }
        assertThat(keywordRetriever.retrieve("HNSW & graph | !memory", 5)).extracting(KeywordRetrieverIT::sourcePath)
                .contains("pgvector.adoc");
    }

    @Test
    void onlySearchesChunksOfTheCurrentEmbeddingModel() {
        RagProperties retiredModel = new RagProperties(properties.corpusDir(), "retired-embedding-model",
                properties.chunking(), properties.retrieval());
        new DocumentIngestionService(vectorStore, documents, new StructureAwareChunker(properties), retiredModel,
                transactionManager)
                .ingest("old.adoc", "old", "= Old\n\nThe zebracorn setting from a retired model.", Origin.CORPUS);

        assertThat(keywordRetriever.retrieve("zebracorn", 10)).isEmpty();
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q verify -Dit.test=KeywordRetrieverIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false`
Expected: compilation FAILURE, `cannot find symbol: class KeywordRetriever`.

- [ ] **Step 3: Implement `KeywordRetriever`**

`src/main/java/com/learnings/rag/retrieval/KeywordRetriever.java`:

```java
package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.ChunkMetadata;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.json.JsonMapper;

/**
 * Lexical retrieval over the generated {@code content_tsv} column (GIN-indexed, {@code english} configuration).
 * {@code websearch_to_tsquery} parses the question (stop words, stemming, "quoted phrases", -negation) and never
 * fails on user input, but joins the terms with AND, so a natural-language question rarely matches any chunk.
 * Replacing {@code &} with {@code |} asks for chunks matching any term; {@code ts_rank_cd} then ranks chunks that
 * match more terms, closer together, higher. Dotted identifiers stay single lexemes, so exact property and class
 * names still match precisely.
 */
@Component
public class KeywordRetriever {

    private static final TypeReference<Map<String, Object>> METADATA = new TypeReference<>() {
    };

    private static final String SEARCH = """
            SELECT id::text AS id, content, metadata::text AS metadata, ts_rank_cd(content_tsv, query) AS rank
            FROM vector_store,
                 replace(websearch_to_tsquery('english', :question)::text, '&', '|')::tsquery AS query
            WHERE content_tsv @@ query AND metadata->>'%s' = :model
            ORDER BY rank DESC, id
            LIMIT :limit
            """.formatted(ChunkMetadata.EMBEDDING_MODEL);

    private final JdbcClient jdbc;
    private final JsonMapper json;
    private final String embeddingModel;

    public KeywordRetriever(JdbcClient jdbc, JsonMapper json, RagProperties properties) {
        this.jdbc = jdbc;
        this.json = json;
        this.embeddingModel = properties.embeddingModel();
    }

    /** Up to {@code limit} chunks, best first; empty when the question has no searchable words. */
    public List<Document> retrieve(String question, int limit) {
        return jdbc.sql(SEARCH)
                .param("question", question)
                .param("model", embeddingModel)
                .param("limit", limit)
                .query((rs, rowNum) -> Document.builder()
                        .id(rs.getString("id"))
                        .text(rs.getString("content"))
                        .metadata(json.readValue(rs.getString("metadata"), METADATA))
                        .score(rs.getDouble("rank"))
                        .build())
                .list();
    }
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `./mvnw -q verify -Dit.test=KeywordRetrieverIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/retrieval/KeywordRetriever.java src/test/java/com/learnings/rag/retrieval/KeywordRetrieverIT.java
git commit -m "feat: keyword retriever over content_tsv with OR semantics and ts_rank_cd"
```

---

### Task 4: Reciprocal rank fusion (M4)

**Files:**
- Create: `src/main/java/com/learnings/rag/retrieval/ReciprocalRankFusion.java`
- Test: `src/test/java/com/learnings/rag/retrieval/ReciprocalRankFusionTest.java`

**Interfaces:**
- Produces: `final class ReciprocalRankFusion` with:
  - `static final int DEFAULT_K = 60`;
  - `static List<Document> fuse(List<List<Document>> rankings, int k, int limit)`, which fuses by Document id, scores with the fused score, sorts best first, and breaks ties by first-seen order.

- [ ] **Step 1: Write the failing test**

`src/test/java/com/learnings/rag/retrieval/ReciprocalRankFusionTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

class ReciprocalRankFusionTest {

    private static Document doc(String id, String retriever) {
        return Document.builder().id(id).text("text " + id).metadata(Map.of("from", retriever)).score(0.5).build();
    }

    private static List<Document> vector() {
        return List.of(doc("a", "vector"), doc("b", "vector"), doc("c", "vector"));
    }

    private static List<Document> keyword() {
        return List.of(doc("c", "keyword"), doc("d", "keyword"));
    }

    @Test
    void aDocumentInBothRankingsAppearsOnceWithSummedScore() {
        List<Document> fused = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10);

        assertThat(fused).extracting(Document::getId).containsExactly("c", "a", "b", "d");
        assertThat(fused.getFirst().getScore()).isCloseTo(1.0 / 63 + 1.0 / 61, within(1e-12));
    }

    @Test
    void tiesKeepTheOrderDocumentsWereFirstSeenIn() {
        // b (vector rank 2) and d (keyword rank 2) both score 1/62; b was seen first.
        List<Document> fused = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10);

        assertThat(fused.get(2).getScore()).isEqualTo(fused.get(3).getScore());
        assertThat(fused).extracting(Document::getId).containsSubsequence("b", "d");
    }

    @Test
    void theLimitCapsTheFusedList() {
        assertThat(ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 2))
                .extracting(Document::getId).containsExactly("c", "a");
    }

    @Test
    void theFirstSeenCopyOfADocumentIsKept() {
        Document fusedC = ReciprocalRankFusion.fuse(List.of(vector(), keyword()), 60, 10).getFirst();

        assertThat(fusedC.getMetadata()).containsEntry("from", "vector");
        assertThat(fusedC.getText()).isEqualTo("text c");
    }

    @Test
    void emptyRankingsFuseToNothing() {
        assertThat(ReciprocalRankFusion.fuse(List.of(List.of(), List.of()), 60, 10)).isEmpty();
    }

    @Test
    void aSingleRankingKeepsItsOrder() {
        assertThat(ReciprocalRankFusion.fuse(List.of(vector()), ReciprocalRankFusion.DEFAULT_K, 10))
                .extracting(Document::getId).containsExactly("a", "b", "c");
    }
}
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q test -Dtest=ReciprocalRankFusionTest`
Expected: compilation FAILURE, `cannot find symbol: class ReciprocalRankFusion`.

- [ ] **Step 3: Implement `ReciprocalRankFusion`**

`src/main/java/com/learnings/rag/retrieval/ReciprocalRankFusion.java`:

```java
package com.learnings.rag.retrieval;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.ai.document.Document;

/**
 * Reciprocal rank fusion (Cormack, Clarke and Büttcher, 2009): {@code score(d) = Σ 1 / (k + rank_i(d))} over the
 * rankings that contain d. Only ranks count, so retrievers with incomparable scores (cosine similarity and
 * {@code ts_rank_cd}) combine without any normalization; {@code k} damps the influence of the very top ranks.
 */
public final class ReciprocalRankFusion {

    /** The constant from the original paper; the spec fixes it at 60. */
    public static final int DEFAULT_K = 60;

    private ReciprocalRankFusion() {
    }

    /**
     * Fuses rankings (each best first) by document id. Returns at most {@code limit} documents, best first, each
     * scored with its fused score; the first-seen copy of a document is kept, and ties keep first-seen order.
     */
    public static List<Document> fuse(List<List<Document>> rankings, int k, int limit) {
        Map<String, Double> scores = new LinkedHashMap<>();
        Map<String, Document> firstSeen = new HashMap<>();
        for (List<Document> ranking : rankings) {
            for (int i = 0; i < ranking.size(); i++) {
                Document document = ranking.get(i);
                scores.merge(document.getId(), 1.0 / (k + i + 1), Double::sum);
                firstSeen.putIfAbsent(document.getId(), document);
            }
        }
        return scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed()) // stable: ties keep first-seen order
                .limit(limit)
                .map(entry -> {
                    Document document = firstSeen.get(entry.getKey());
                    return Document.builder()
                            .id(document.getId())
                            .text(document.getText())
                            .metadata(document.getMetadata())
                            .score(entry.getValue())
                            .build();
                })
                .toList();
    }
}
```

- [ ] **Step 4: Run it and watch it pass**

Run: `./mvnw -q test -Dtest=ReciprocalRankFusionTest`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/retrieval/ReciprocalRankFusion.java src/test/java/com/learnings/rag/retrieval/ReciprocalRankFusionTest.java
git commit -m "feat: reciprocal rank fusion (k=60) with first-seen tie-breaking"
```

---

### Task 5: Retrieval modes and parallel hybrid search (M4)

**Files:**
- Create: `src/main/java/com/learnings/rag/retrieval/RetrievalMode.java`
- Modify: `src/main/java/com/learnings/rag/config/RagProperties.java`, `src/main/resources/application.yml`, `src/main/java/com/learnings/rag/retrieval/{RetrievalOptions, PipelineTrace, RetrievalPipeline}.java`, `src/main/java/com/learnings/rag/eval/EvalConfig.java`
- Test: `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineTest.java` (new), `RetrievalPipelineIT.java`, `RetrievalOptionsTest.java`, `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`

**Interfaces:**
- Consumes: `KeywordRetriever.retrieve(String, int)` (Task 3); `ReciprocalRankFusion.fuse` and `DEFAULT_K` (Task 4).
- Produces:
  - `enum RetrievalMode { VECTOR, KEYWORD, HYBRID }`.
  - `RagProperties.Retrieval(int topK, double similarityThreshold, RetrievalMode mode, int candidates)`, defaulting to `VECTOR` and `20`.
  - `RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates)` with `from(RagProperties)`, `withTopK(int)` and `withMode(RetrievalMode)`.
  - `PipelineTrace(List<Stage> stages, long totalMillis)`, plus a 1-argument constructor in which the total is the sum of the stages.
  - `RetrievalPipeline(VectorRetriever, KeywordRetriever, RagProperties)` with:
    - stage `vector` for VECTOR;
    - stage `keyword` for KEYWORD;
    - stages `vector`, `keyword`, `fusion` for HYBRID.
  - `EvalConfig.all(RagProperties)` stays M3's single `vector` config, now built with `withMode(VECTOR)`; Task 7 adds `keyword` and `hybrid`.

- [ ] **Step 1: Write the failing tests**

Replace `src/test/java/com/learnings/rag/retrieval/RetrievalOptionsTest.java` with:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import com.learnings.rag.config.RagProperties;

class RetrievalOptionsTest {

    private static RagProperties properties(int topK, double threshold, RetrievalMode mode, int candidates) {
        return new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(topK, threshold, mode, candidates));
    }

    @Test
    void defaultsComeFromTheRetrievalProperties() {
        assertThat(RetrievalOptions.from(properties(5, 0.2, RetrievalMode.HYBRID, 20)))
                .isEqualTo(new RetrievalOptions(5, 0.2, RetrievalMode.HYBRID, 20));
    }

    @Test
    void withTopKAndWithModeChangeOnlyThatField() {
        RetrievalOptions options = new RetrievalOptions(5, 0.2, RetrievalMode.VECTOR, 20);

        assertThat(options.withTopK(10)).isEqualTo(new RetrievalOptions(10, 0.2, RetrievalMode.VECTOR, 20));
        assertThat(options.withMode(RetrievalMode.KEYWORD)).isEqualTo(new RetrievalOptions(5, 0.2, RetrievalMode.KEYWORD, 20));
    }

    @Test
    void topKAndCandidatesMustBePositiveAndModeIsRequired() {
        assertThatThrownBy(() -> new RetrievalOptions(0, 0.0, RetrievalMode.VECTOR, 20)).hasMessageContaining("topK");
        assertThatThrownBy(() -> new RetrievalOptions(5, 0.0, RetrievalMode.VECTOR, 0)).hasMessageContaining("candidates");
        assertThatThrownBy(() -> new RetrievalOptions(5, 0.0, null, 20)).hasMessageContaining("mode");
    }
}
```

`src/test/java/com/learnings/rag/retrieval/RetrievalPipelineTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import com.learnings.rag.config.RagProperties;

class RetrievalPipelineTest {

    private final VectorRetriever vectorRetriever = mock(VectorRetriever.class);
    private final KeywordRetriever keywordRetriever = mock(KeywordRetriever.class);
    private final RetrievalPipeline pipeline = new RetrievalPipeline(vectorRetriever, keywordRetriever,
            new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                    new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20)));

    private static Document doc(String id) {
        return Document.builder().id(id).text("text " + id).score(0.5).build();
    }

    @Test
    void aFailingRetrieverFailsTheHybridSearchWithItsOwnException() {
        when(vectorRetriever.retrieve(any(), any())).thenThrow(new IllegalStateException("embedding service unavailable"));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of(doc("k1")));

        assertThatThrownBy(() -> pipeline.retrieve("question"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("embedding service unavailable");
    }

    @Test
    void hybridStillAnswersWhenKeywordSearchFindsNothing() {
        when(vectorRetriever.retrieve(any(), any())).thenReturn(List.of(doc("v1"), doc("v2")));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of());

        RetrievalResult result = pipeline.retrieve("how do I do it?");

        assertThat(result.documents()).extracting(Document::getId).containsExactly("v1", "v2");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("vector", "keyword", "fusion");
    }

    @Test
    void hybridAsksEachRetrieverForTheCandidateCountAndReturnsTopK() {
        when(vectorRetriever.retrieve(any(), any())).thenReturn(
                List.of(doc("a"), doc("b"), doc("c"), doc("d"), doc("e"), doc("f")));
        when(keywordRetriever.retrieve(anyString(), anyInt())).thenReturn(List.of(doc("f"), doc("g")));

        RetrievalResult result = pipeline.retrieve("question");

        assertThat(result.documents()).hasSize(5).extracting(Document::getId).doesNotHaveDuplicates();
        assertThat(result.documents().getFirst().getId()).isEqualTo("f"); // in both rankings
    }
}
```

In `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineIT.java`, add these tests after `topKCanBeOverriddenPerCall`:

```java
    @Test
    void keywordModeFindsAnExactIdentifier() {
        RetrievalResult result = pipeline.retrieve("spring.ai.vectorstore.pgvector.index-type",
                RetrievalOptions.from(properties).withMode(RetrievalMode.KEYWORD));

        assertThat(result.documents().getFirst().getMetadata())
                .containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc")
                .containsEntry(ChunkMetadata.BREADCRUMB, "Configuration properties");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("keyword");
    }

    @Test
    void hybridModeFusesBothRetrieversIntoTopK() {
        RetrievalResult result = pipeline.retrieve("Which index type is HNSW and how does it build its graph?",
                RetrievalOptions.from(properties).withMode(RetrievalMode.HYBRID));

        assertThat(result.documents()).isNotEmpty().hasSizeLessThanOrEqualTo(5)
                .extracting(Document::getId).doesNotHaveDuplicates();
        assertThat(result.documents().getFirst().getMetadata()).containsEntry(ChunkMetadata.SOURCE_PATH, "pgvector.adoc");
        assertThat(result.documents()).allSatisfy(document ->
                assertThat(document.getScore()).isLessThanOrEqualTo(2.0 / (ReciprocalRankFusion.DEFAULT_K + 1)));
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("vector", "keyword", "fusion");
        assertThat(result.trace().totalMillis()).isNotNegative();
    }
```

Also add the import `org.springframework.ai.document.Document` to that file if it is missing.

In `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`, change `new RetrievalOptions(10, 0.0)` to `new RetrievalOptions(10, 0.0, RetrievalMode.VECTOR, 20)`, and add the import `com.learnings.rag.retrieval.RetrievalMode`.

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest='RetrievalOptionsTest,RetrievalPipelineTest'`
Expected: compilation FAILURE, `cannot find symbol: class RetrievalMode`.

- [ ] **Step 3: Implement the modes**

`src/main/java/com/learnings/rag/retrieval/RetrievalMode.java`:

```java
package com.learnings.rag.retrieval;

/** Which retrievers answer a question. HYBRID runs vector and keyword search and fuses them with RRF. */
public enum RetrievalMode {
    VECTOR, KEYWORD, HYBRID
}
```

In `src/main/java/com/learnings/rag/config/RagProperties.java`, add the import `com.learnings.rag.retrieval.RetrievalMode` and replace the `Retrieval` record and its javadoc with:

```java
    /**
     * @param topK number of chunks handed to the model
     * @param similarityThreshold minimum cosine similarity for vector search; 0 keeps every positive similarity
     * @param mode which retrievers run: VECTOR, KEYWORD, or HYBRID (both, fused with reciprocal rank fusion)
     * @param candidates in HYBRID mode, how many chunks each retriever contributes before fusion
     */
    public record Retrieval(@DefaultValue("5") int topK,
            @DefaultValue("0.0") double similarityThreshold,
            @DefaultValue("VECTOR") RetrievalMode mode,
            @DefaultValue("20") int candidates) {
    }
```

In `src/main/resources/application.yml`, replace

```yaml
  retrieval:
    top-k: 5
    similarity-threshold: 0.0
```

with

```yaml
  retrieval:
    top-k: 5
    similarity-threshold: 0.0
    mode: vector          # vector | keyword | hybrid; chosen by the M4 eval (see eval/README.md)
    candidates: 20        # hybrid: chunks each retriever contributes before fusion
```

Replace `src/main/java/com/learnings/rag/retrieval/RetrievalOptions.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.Objects;

import com.learnings.rag.config.RagProperties;

/**
 * Per-call retrieval settings. The defaults come from {@code rag.retrieval.*}; the eval harness overrides them, so
 * one run can compare several configurations and retrieve deeper than the chat endpoint does.
 *
 * @param topK number of chunks to return
 * @param similarityThreshold minimum cosine similarity for vector search; 0 keeps every positive similarity
 * @param mode which retrievers run
 * @param candidates in HYBRID mode, how many chunks each retriever contributes before fusion
 */
public record RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
        if (candidates < 1) {
            throw new IllegalArgumentException("candidates must be at least 1, was " + candidates);
        }
        Objects.requireNonNull(mode, "mode");
    }

    public static RetrievalOptions from(RagProperties properties) {
        RagProperties.Retrieval retrieval = properties.retrieval();
        return new RetrievalOptions(retrieval.topK(), retrieval.similarityThreshold(), retrieval.mode(),
                retrieval.candidates());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates);
    }

    public RetrievalOptions withMode(RetrievalMode mode) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates);
    }
}
```

Replace `src/main/java/com/learnings/rag/retrieval/PipelineTrace.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/**
 * What each retrieval stage did and how long it took; feeds the done event and the sources' per-stage scores.
 * Stages can run in parallel (HYBRID runs vector and keyword search at once), so {@code totalMillis} is the
 * wall-clock time of the whole retrieval, not the sum of the stages.
 */
public record PipelineTrace(List<Stage> stages, long totalMillis) {

    /** For stages that ran one after another: the total is their sum. */
    public PipelineTrace(List<Stage> stages) {
        this(stages, stages.stream().mapToLong(Stage::elapsedMillis).sum());
    }

    public record Stage(String name, long elapsedMillis, List<Hit> hits) {
    }

    public record Hit(String id, String sourcePath, String breadcrumb, Double score) {

        static Hit of(Document document) {
            return new Hit(document.getId(),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                    Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                    document.getScore());
        }
    }
}
```

Replace `src/main/java/com/learnings/rag/retrieval/RetrievalPipeline.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;

/**
 * Retrieves chunks for a question in the configured {@link RetrievalMode}. HYBRID runs vector and keyword search in
 * parallel on virtual threads and fuses their candidates with reciprocal rank fusion. Every stage is traced. Later
 * milestones add query rewriting, multi-query expansion and reranking here.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final KeywordRetriever keywordRetriever;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, KeywordRetriever keywordRetriever,
            RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.keywordRetriever = keywordRetriever;
        this.defaults = RetrievalOptions.from(properties);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Timed> stages = switch (options.mode()) {
            case VECTOR -> List.of(timed("vector", () -> vectorRetriever.retrieve(new Query(question), options)));
            case KEYWORD -> List.of(timed("keyword", () -> keywordRetriever.retrieve(question, options.topK())));
            case HYBRID -> hybrid(question, options);
        };
        PipelineTrace trace = new PipelineTrace(stages.stream().map(Timed::stage).toList(), millisSince(start));
        return new RetrievalResult(stages.getLast().documents(), trace);
    }

    private List<Timed> hybrid(String question, RetrievalOptions options) {
        int candidates = Math.max(options.candidates(), options.topK());
        Timed vector;
        Timed keyword;
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Timed> vectorSearch = executor.submit(() -> timed("vector",
                    () -> vectorRetriever.retrieve(new Query(question), options.withTopK(candidates))));
            Future<Timed> keywordSearch = executor.submit(() -> timed("keyword",
                    () -> keywordRetriever.retrieve(question, candidates)));
            vector = join(vectorSearch);
            keyword = join(keywordSearch);
        }
        Timed fusion = timed("fusion", () -> ReciprocalRankFusion.fuse(
                List.of(vector.documents(), keyword.documents()), ReciprocalRankFusion.DEFAULT_K, options.topK()));
        return List.of(vector, keyword, fusion);
    }

    /** Waits for a parallel search; its own exception is rethrown as is, not wrapped. */
    private static Timed join(Future<Timed> search) {
        try {
            return search.get();
        }
        catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while retrieving", e);
        }
        catch (ExecutionException e) {
            if (e.getCause() instanceof RuntimeException runtime) {
                throw runtime;
            }
            throw new IllegalStateException(e.getCause());
        }
    }

    private record Timed(String name, List<Document> documents, long millis) {

        PipelineTrace.Stage stage() {
            return new PipelineTrace.Stage(name, millis, documents.stream().map(PipelineTrace.Hit::of).toList());
        }
    }

    private static Timed timed(String name, Supplier<List<Document>> search) {
        long start = System.nanoTime();
        List<Document> documents = search.get();
        return new Timed(name, documents, millisSince(start));
    }

    private static long millisSince(long start) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
    }
}
```

In `src/main/java/com/learnings/rag/eval/EvalConfig.java`, replace the body of `all(...)` so that it pins the mode explicitly:

```java
        return List.of(new EvalConfig("vector", RetrievalOptions.from(properties)
                .withTopK(RetrievalMetrics.MRR_K).withMode(RetrievalMode.VECTOR)));
```

Add the import `com.learnings.rag.retrieval.RetrievalMode`.

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./mvnw -q verify -Dtest='RetrievalOptionsTest,RetrievalPipelineTest,ReportWriterTest,AnswerServiceTest' -Dit.test=RetrievalPipelineIT`
Expected: PASS (3 + 3 + 7 + 5 unit tests, 6 ITs).

- [ ] **Step 5: Run every test**

Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS.

- [ ] **Step 6: Commit**

```bash
git add src/main/java/com/learnings/rag src/main/resources/application.yml src/test/java/com/learnings/rag
git commit -m "feat: retrieval modes VECTOR, KEYWORD and HYBRID (parallel search fused with RRF)"
```

---

### Task 6: Per-stage scores in chat sources (M4)

**Files:**
- Modify: `src/main/java/com/learnings/rag/generation/{SourceRef, ChatEvent, AnswerService}.java`, `src/main/resources/static/app.js`
- Test: `src/test/java/com/learnings/rag/generation/AnswerServiceTest.java`

**Interfaces:**
- Consumes: `RetrievalResult`, `PipelineTrace` and its stages and hits (Task 5).
- Produces:
  - `SourceRef` gains `Map<String, Double> scores`, from stage name to score, in stage order. The 6-argument constructor stays and means no stage scores.
  - `ChatEvent.Sources.from(RetrievalResult)`.

- [ ] **Step 1: Write the failing test**

In `src/test/java/com/learnings/rag/generation/AnswerServiceTest.java`, add this test after `streamsSourcesThenTokensThenUsage`:

```java
    @Test
    void sourcesCarryTheScoreOfEveryStageThatReturnedTheChunk() {
        RetrievalResult retrieval = oneChunk();
        Document chunk = retrieval.documents().getFirst();
        PipelineTrace trace = new PipelineTrace(List.of(
                new PipelineTrace.Stage("vector", 3, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.61))),
                new PipelineTrace.Stage("keyword", 2, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.08))),
                new PipelineTrace.Stage("fusion", 0, List.of(new PipelineTrace.Hit(chunk.getId(), "pgvector.adoc", "Indexes", 0.0325)))),
                4);
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(retrieval.documents(), trace));

        StepVerifier.create(service(new StubChatModel("ok [1]")).answer("q"))
                .assertNext(event -> assertThat(event).isInstanceOfSatisfying(ChatEvent.Sources.class,
                        sources -> assertThat(sources.sources().getFirst().scores())
                                .containsExactly(Map.entry("vector", 0.61), Map.entry("keyword", 0.08),
                                        Map.entry("fusion", 0.0325))))
                .thenConsumeWhile(event -> true)
                .verifyComplete();
    }
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q test -Dtest=AnswerServiceTest`
Expected: compilation FAILURE, `cannot find symbol: method scores()`.

- [ ] **Step 3: Implement per-stage scores**

Replace `src/main/java/com/learnings/rag/generation/SourceRef.java` with:

```java
package com.learnings.rag.generation;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;
import com.learnings.rag.retrieval.PipelineTrace;

/**
 * @param n the citation number the model uses ([n])
 * @param score what the chunk was ranked by: cosine similarity (vector), ts_rank_cd (keyword) or the fused RRF score
 * @param scores the chunk's score in each retrieval stage that returned it, in stage order,
 *        e.g. {vector=0.61, keyword=0.08, fusion=0.0325}
 */
public record SourceRef(int n, String sourcePath, String title, String breadcrumb, Double score, String text,
        Map<String, Double> scores) {

    public SourceRef(int n, String sourcePath, String title, String breadcrumb, Double score, String text) {
        this(n, sourcePath, title, breadcrumb, score, text, Map.of());
    }

    static SourceRef of(int n, Document document, PipelineTrace trace) {
        Map<String, Double> scores = new LinkedHashMap<>();
        for (PipelineTrace.Stage stage : trace.stages()) {
            stage.hits().stream()
                    .filter(hit -> hit.id().equals(document.getId()) && hit.score() != null)
                    .findFirst()
                    .ifPresent(hit -> scores.put(stage.name(), hit.score()));
        }
        var metadata = document.getMetadata();
        return new SourceRef(n,
                Objects.toString(metadata.get(ChunkMetadata.SOURCE_PATH), ""),
                Objects.toString(metadata.get(ChunkMetadata.TITLE), ""),
                Objects.toString(metadata.get(ChunkMetadata.BREADCRUMB), ""),
                document.getScore(),
                document.getText(),
                Collections.unmodifiableMap(scores));
    }
}
```

In `src/main/java/com/learnings/rag/generation/ChatEvent.java`, replace the `from` method of `Sources` with the following, and add the import `com.learnings.rag.retrieval.RetrievalResult`:

```java
        static Sources from(RetrievalResult retrieval) {
            List<Document> documents = retrieval.documents();
            return new Sources(IntStream.range(0, documents.size())
                    .mapToObj(i -> SourceRef.of(i + 1, documents.get(i), retrieval.trace()))
                    .toList());
        }
```

In `src/main/java/com/learnings/rag/generation/AnswerService.java`, change `ChatEvent sources = ChatEvent.Sources.from(documents);` to `ChatEvent sources = ChatEvent.Sources.from(retrieval);`.

In `src/main/resources/static/app.js`:

1. In `renderSources`, replace `textContent: s.score == null ? '' : `similarity ${s.score.toFixed(3)}`` with `textContent: scoreText(s)`.
2. Add this function directly above `function showSource(n) {`:

```js
// The score of each stage that returned the chunk; vector search is shown as "similarity".
function scoreText(s) {
  const entries = Object.entries(s.scores || {});
  if (entries.length === 0) return s.score == null ? '' : `score ${s.score.toFixed(3)}`;
  return entries.map(([stage, value]) => `${stage === 'vector' ? 'similarity' : stage} ${value.toFixed(3)}`).join(' · ');
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./mvnw -q test -Dtest='AnswerServiceTest,ChatControllerTest' && node --check src/main/resources/static/app.js`
Expected: PASS (6 + 4 tests); the JS syntax check is clean.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/generation src/main/resources/static/app.js src/test/java/com/learnings/rag/generation
git commit -m "feat: chat sources carry each retrieval stage's score; UI shows them"
```

---

### Task 7: Compare vector, keyword and hybrid; choose the default (M4)

**Files:**
- Modify: `src/main/java/com/learnings/rag/eval/EvalConfig.java`, `src/test/java/com/learnings/rag/eval/EvalRunnerIT.java`
- Modify (docs): `eval/README.md`, `README.md`, `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`; `src/main/resources/application.yml` only if the rule below picks HYBRID

**Interfaces:**
- Consumes: `RetrievalMode` and `RetrievalOptions.withMode` (Task 5); per-tag summaries (Task 1); the identifier items (Task 2).
- Produces: `EvalConfig.all(RagProperties)` returns `vector`, `keyword` and `hybrid`, all with top 10 and 20 candidates.

- [ ] **Step 1: Write the failing test**

In `src/test/java/com/learnings/rag/eval/EvalRunnerIT.java`, in `scoresEveryQuestionAndSummarizesTheConfig`, replace

```java
        assertThat(report.configs()).singleElement().satisfies(config -> {
```

with

```java
        assertThat(report.configs()).extracting(EvalReport.ConfigResult::name)
                .containsExactly("vector", "keyword", "hybrid");
        assertThat(report.configs()).allSatisfy(config -> assertThat(config.items()).hasSize(2));
        assertThat(report.configs().getFirst()).satisfies(config -> {
```

- [ ] **Step 2: Run it and watch it fail**

Run: `./mvnw -q verify -Dit.test=EvalRunnerIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false`
Expected: FAIL. The report has only `vector`: "Expecting actual: ["vector"] to contain exactly ["vector", "keyword", "hybrid"]".

- [ ] **Step 3: Add the configurations**

In `src/main/java/com/learnings/rag/eval/EvalConfig.java`, replace `all(...)` and its javadoc with:

```java
    /** The configurations every run compares, all retrieving the top 10. M5–M6 add multi-query and rerank variants. */
    public static List<EvalConfig> all(RagProperties properties) {
        RetrievalOptions base = RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K);
        return List.of(
                new EvalConfig("vector", base.withMode(RetrievalMode.VECTOR)),
                new EvalConfig("keyword", base.withMode(RetrievalMode.KEYWORD)),
                new EvalConfig("hybrid", base.withMode(RetrievalMode.HYBRID)));
    }
```

- [ ] **Step 4: Run it and watch it pass, then run every test**

Run: `./mvnw -q verify -Dit.test=EvalRunnerIT -Dtest=skip -Dsurefire.failIfNoSpecifiedTests=false && ./mvnw -q verify`
Expected: PASS; then all unit tests and ITs PASS.

- [ ] **Step 5: Commit**

```bash
git add src/main/java/com/learnings/rag/eval/EvalConfig.java src/test/java/com/learnings/rag/eval/EvalRunnerIT.java
git commit -m "feat: eval compares vector, keyword and hybrid retrieval"
```

- [ ] **Step 6: Run the M4 eval (calls OpenAI for query embeddings)**

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval`
Expected:
- three log lines (`vector: …`, `keyword: …`, `hybrid: …`), then `Report written to eval/reports/<time>.md`;
- the report has three summary rows and a `## By tag` table with `identifier` and `untagged` rows for each config.

Read the report: the summary, the by-tag table, and the misses for each config. Note where keyword and hybrid win or lose against vector, and on which items.

- [ ] **Step 7: Apply the default-mode rule (spec amendment 24)**

Let N = 53, the number of golden items. **HYBRID becomes the default** only if both of these hold:
- its overall MRR@10 is ≥ vector's;
- its overall hit@5 is ≥ vector's hit@5 − 1/N.

Otherwise VECTOR stays.

- **If HYBRID wins:**
  1. Change `mode: vector` to `mode: hybrid` in `src/main/resources/application.yml`.
  2. Run `./mvnw -q verify`. The tests pin their modes or use the defaults; investigate any test that changes outcome, and record a ruling.
  3. Start the app (`./mvnw spring-boot:run`) and ask "How do I configure the HNSW index for PGvector?" in the browser. The sources must show `similarity … · keyword … · fusion …`.
  4. Because the sources panel now looks different, re-capture `docs/images/01-ask-cited-answer.png` and `02-citation-opens-source.png` the same way as before (1280×860 viewport, full page for 01), then stop the app.
- **If VECTOR stays:** leave `application.yml` unchanged.

- [ ] **Step 8: Record the results**

In `eval/README.md`, add a section `## M4: vector vs keyword vs hybrid` directly above `## Baseline` containing:
- the report's run-info table, summary table and by-tag table, copied verbatim, with the report path;
- one paragraph on the default-mode decision, giving the rule's numbers;
- 2–4 bullets on what the per-tag and per-item results show, for example where keyword search wins (identifiers) and loses (paraphrased questions).

In `README.md`:
- change the M4 Status row to `✅ done` and M5 to `next`;
- under **Evaluation**, replace the baseline paragraph and table with the M4 summary table, headed `**M4 comparison (53 questions, 2026-10-06):**`, followed by one sentence on the chosen default;
- under **Configuration**, add rows for `rag.retrieval.mode` (default and why) and `rag.retrieval.candidates` (`20`).

In the spec's amendment 24, append the outcome in one sentence: "Outcome (M4 report <time>): <mode> chosen; hybrid MRR@10 x vs vector y, hit@5 a vs b."

- [ ] **Step 9: Commit**

```bash
git add eval/README.md README.md docs/superpowers/specs/2026-10-05-rag-pipeline-design.md src/main/resources/application.yml docs/images
git commit -m "eval: M4 comparison of vector, keyword and hybrid retrieval; default mode chosen by the rule"
```

---

## After this plan

M4 is complete when Task 7 Step 8 records the comparison and the default mode.

Next is **M5: query rewriting and multi-query expansion**. It adds:
- `RewriteQueryTransformer` and `MultiQueryExpander` on a separate utility `ChatClient` (`AiConfig`, deferred from M0–M2);
- retrieval for every query variant, in parallel on virtual threads;
- RRF across the variants;
- `hybrid+multiquery` in `EvalConfig.all`.

The report's latency columns will show the cost of the extra LLM calls. M5 gets its own plan.
