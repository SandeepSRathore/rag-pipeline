# RAG Pipeline M7: Generation Evals, `/api/retrieve` and the Debug Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the *answers*, not just retrieval. For every golden question:
- run the real chat path;
- judge the answer for faithfulness, relevancy and correctness with Spring AI's evaluators;
- check its citations, and classify refusals.

Also add a retrieval-only debug endpoint and a trace panel in the UI.

**Architecture:**
- **Reranker:** a reply that skips a candidate is a failure (the M6 review item).
- **`generation` package:**
  - `CitationValidator` checks `[n]` citations;
  - `AnswerService.isRefusal` recognises refusals;
  - chat can stream a `trace` event on request.
- **`retrieval` package:** `RetrievalController` serves `POST /api/retrieve`.
- **`eval` package:**
  - `AnswerJudge` wraps `FactCheckingEvaluator` and `RelevancyEvaluator` on the utility model;
  - `GenerationEvalRunner` runs every golden question through `AnswerService` and judges the result;
  - `--rag.eval.generation=true` adds a Generation section to the eval report.
- **UI:** a Debug toggle renders the `PipelineTrace`.

**Tech Stack:** Java 25, Spring Boot 4.1.1, Spring AI 2.0.1 (`FactCheckingEvaluator`, `RelevancyEvaluator`, `EvaluationRequest`), OpenAI `gpt-5-mini` for answers and `gpt-4.1-mini` as the judge, JUnit 5, Mockito, Testcontainers, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`. This plan covers milestone M7: the Evaluation harness's Generation bullet, `RetrievalController`, the UI "Debug" toggle and end-to-end check 4. It adds amendments 38–42.

## Global Constraints

- Java 25; Spring Boot `4.1.1`; `spring-ai-bom` `2.0.1`; `./mvnw`. Package `com.learnings.rag`; `rag.*` configuration records.
- Jackson 3 only. Prompts are passed as Message objects or through Spring AI's own evaluators, never as our own template strings.
- The utility client (`gpt-4.1-mini`, temperature 0) is separate from the answer client (`OPENAI_CHAT_MODEL`, default `gpt-5-mini`). The judge uses the utility client, so no model grades its own answers.
- `*Test` = unit tests without Docker; `*IT` = Testcontainers ITs. Tests never call OpenAI; `StubChatModel` stands in.
- The UI renders model and document text as text nodes only, never as HTML.
- Learning project: no auth or rate limits. App on `127.0.0.1:8081`. Work on branch `m7-generation-evals`.

**Spec amendments made while planning (also recorded in the spec):**
38. **A rerank reply that skips a candidate is a failure.** This supersedes amendment 33's "unrated candidates score 0".
    - The fused order is kept and no minimum applies.
    - Why: an unrated candidate scored 0 would be dropped by the minimum, so a partial reply could refuse an answerable question unnoticed (M6 review).
    - It never happened in M6's 292 rerank calls. Fallbacks are already counted in the report.
39. **Generation eval** (`--rag.eval.generation=true` with the `eval` profile, after the retrieval configs):
    - **Path:** every golden question goes through `AnswerService`, the real chat path with the chat defaults (hybrid, top 5, rerank on, min-score 6), using the answer model.
    - **Outcome per question:**
      - `REFUSED_BY_RETRIEVAL`: no sources;
      - `REFUSED_BY_MODEL`: the answer contains the prompt's refusal sentence or the no-sources sentence;
      - `FAILED`: an error event, or an empty answer;
      - `ANSWERED`.
    - **Only answered questions are judged.** Probe while planning: both judges mark a refusal as unsupported.
    - **The judge is Spring AI's evaluators on the utility model:**
      - **faithfulness:** `FactCheckingEvaluator`, the answer as the claim, its sources as the document;
      - **relevancy:** `RelevancyEvaluator` (question, answer, sources);
      - **correctness:** `FactCheckingEvaluator` with the golden `referenceAnswer` as the claim and the answer as the document. Answerable questions only.
    - **Probe while planning:**
      - `gpt-4.1-mini` replied exactly `yes`/`No`/`YES`/`NO` to the default prompts, with correct verdicts on 6 cases;
      - Spring AI passes only an exact `yes` (any case), so a decorated reply ("Yes.") counts as a fail;
      - a judge exception is counted as an error, never a fail.
    - **No decision rule:** M7 records a baseline.
40. **`CitationValidator`:**
    - A citation is `[n]` outside code. Fenced blocks and inline code are skipped, unlike the UI's chips, where `parts[1]` becomes a chip.
    - An answer's citations are valid when it cites at least one source and every cited number is within 1..sources.
    - The rate is reported over answered questions.
41. **`POST /api/retrieve`:** retrieval only, never an answer-model call.
    - **Request:** `{question, mode?, topK?, rewrite?, queryVariants?, rerank?, minScore?}`.
    - **Defaults and bounds:** missing fields take their `rag.retrieval.*` defaults. Bounds: `topK` 1–20, `queryVariants` 0–5, `minScore` 0–10, question at most 2,000 characters. Out-of-range values or an unknown mode → 400.
    - **Response:** `{chunks: [{rank, id, sourcePath, breadcrumb, score, text}], trace}`.
42. **Debug trace:** `POST /api/chat` accepts `"debug": true` and then streams a `trace` event, the `PipelineTrace`, right after `sources`. Without it the protocol is unchanged. The UI's Debug toggle renders it.

## Review Focus

Inputs the spec implies but doesn't spell out, most likely to bite first. Each one gets a test in the task that owns it:

1. **A refusal phrased differently** must be classified as a refusal, not judged as an unfaithful answer. Variants: a prefix such as "Direct answer: I couldn't find this…", a curly apostrophe, or a different case.
   - Covered by `refusalsAreRecognisedHoweverTheModelPhrasesThem` (Task 2).
   - Also by `refusalsAreClassifiedNotJudged` (Task 5).
2. **Numbers in code** (`parts[1]`, `array[0]`, an unterminated fence) must not count as citations. A `[7]` with 5 sources, or an answer that cites nothing, is invalid.
   - Covered by `CitationValidatorTest` (Task 2).
3. **Braces, Markdown and code** in sources or answers must reach the judges intact, through Spring AI's template rendering.
   - Covered by `bracesAndCodeReachTheJudgesVerbatim` (Task 5).
4. **A judge reply that isn't a plain yes or no:** a decorated "Yes." counts as a fail (Spring AI's rule, pinned so an upgrade that changes it is noticed). An exception is an error, counted apart.
   - Covered by `aDecoratedYesFails` and `aFailingJudgeIsAnErrorNotAFail` (Task 5).
5. **`/api/retrieve` with a blank question, out-of-range numbers or an unknown mode** must answer 400, never 500, and never do unbounded work.
   - Covered by `outOfRangeOrUnknownSettingsAreRejected` (Task 3).

---

## File map

```
src/main/java/com/learnings/rag/
  retrieval/LlmReranker.java                    partial reply → failure (modify)
  retrieval/RetrievalController.java, RetrieveRequest.java, RetrieveResponse.java   POST /api/retrieve (new)
  generation/CitationValidator.java             [n] checks (new)
  generation/AnswerService.java                 isRefusal, PROMPT_REFUSAL, answer(question, debug) (modify)
  generation/ChatEvent.java, ChatRequest.java, ChatController.java   trace event, debug flag (modify)
  eval/AnswerJudge.java, GenerationEvalRunner.java, GenerationReport.java   generation eval (new)
  eval/EvalProperties.java, EvalReport.java, EvalCommand.java, ReportWriter.java   flag + report section (modify)
src/main/resources/static/{index.html, app.js, app.css}   Debug toggle + trace panel
src/test/java/com/learnings/rag/
  generation/CitationValidatorTest.java, retrieval/RetrieveRequestTest.java, retrieval/RetrievalControllerIT.java,
  eval/AnswerJudgeTest.java, eval/GenerationEvalRunnerTest.java (new)
  retrieval/LlmRerankerTest.java, generation/{AnswerServiceTest, ChatControllerTest}.java, eval/ReportWriterTest.java (modify)
eval/README.md, README.md, spec, docs/images/07-debug-trace.png
```

---

### Task 1: A rerank reply that skips a candidate is a failure (M7, deferred from M6)

**Files:**
- Modify: `src/main/java/com/learnings/rag/retrieval/LlmReranker.java`
- Test: `src/test/java/com/learnings/rag/retrieval/LlmRerankerTest.java`
- Modify: `README.md` (one bullet)

**Interfaces:**
- Consumes: nothing new.
- Produces: `LlmReranker.rerank` returns `Optional.empty()` when any candidate is unrated. The signature is unchanged.

- [ ] **Step 1: Write the failing test and update the messy-ratings test**

In `src/test/java/com/learnings/rag/retrieval/LlmRerankerTest.java`:

1. In `messyRatingsAreCleanedNotTrusted`, rate candidate 4 too. Replace its comment and reply with:

```java
        // A string id with a decimal score; a score above 10; a repeated id (the first rating counts); unknown ids;
        // a negative score.
        StubChatModel model = new StubChatModel("{\"ratings\": [{\"id\": \"2\", \"score\": 7.5}, {\"id\": 1, \"score\": 14},"
                + " {\"id\": 2, \"score\": 1}, {\"id\": 0, \"score\": 10}, {\"id\": 9, \"score\": 10},"
                + " {\"id\": 3, \"score\": -2}, {\"id\": 4, \"score\": 0}]}");
```

(The expected order `a, b, c, d` and scores `10.0, 7.5, 0.0, 0.0` stay as they are.)

2. Add:

```java
    @Test
    void aReplyThatSkipsACandidateIsAFailure() {
        // An unrated candidate would score 0 and be dropped by the minimum score, so a partial reply could refuse an
        // answerable question. Keeping the fused order is the safe side.
        StubChatModel model = new StubChatModel("{\"ratings\": [{\"id\": 1, \"score\": 9}, {\"id\": 2, \"score\": 8}]}");

        assertThat(reranker(model).rerank(QUESTION, candidates())).isEmpty();
    }
```

- [ ] **Step 2: Run the tests to verify the new one fails**

Run: `./mvnw -q test -Dtest=LlmRerankerTest`
Expected: FAIL in `aReplyThatSkipsACandidateIsAFailure` only. The current code returns the three candidates with candidate 3 at 0.

- [ ] **Step 3: Treat partial replies as failures**

In `src/main/java/com/learnings/rag/retrieval/LlmReranker.java`:
1. In the class javadoc, replace `scores are clamped to 0–{@value #MAX_SCORE}, and a chunk left unrated scores 0. A failed` with `scores are clamped to 0–{@value #MAX_SCORE}. A failed`, and replace `or a reply without one usable rating, comes back empty` with `or a reply that leaves any candidate unrated, comes back empty`.
2. Replace

```java
        if (scores.size() < candidates.size()) {
            log.warn("Reranking rated {} of {} candidates; the rest score 0", scores.size(), candidates.size());
        }
```

with

```java
        if (scores.size() < candidates.size()) {
            // An unrated candidate scored 0 would be dropped by the minimum score: a partial reply could refuse an
            // answerable question. The fused order is the safe side.
            log.warn("Reranking rated {} of {} candidates; keeping the fused order", scores.size(), candidates.size());
            return Optional.empty();
        }
```

3. In the loop that builds `rated`, replace `.score(scores.getOrDefault(i + 1, 0.0))` with `.score(scores.get(i + 1))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest=LlmRerankerTest`
Expected: PASS, 11 tests. `thePromptNumbersThePassagesInOrderAndNeverShowsChunkIds` now logs "rated 1 of 2 … keeping the fused order"; it asserts only the prompt, so it still passes.

- [ ] **Step 5: Update the README bullet**

In `README.md`, under **Rerank**, replace `- The ratings are cleaned: clamped to 0–10, deduplicated, and unrated candidates score 0.` with `- The ratings are cleaned: clamped to 0–10 and deduplicated. A reply that skips a candidate counts as a failure.`

- [ ] **Step 6: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add src/main/java/com/learnings/rag/retrieval/LlmReranker.java src/test/java/com/learnings/rag/retrieval/LlmRerankerTest.java README.md
git commit -m "fix: a rerank reply that skips a candidate keeps the fused order instead of scoring it 0"
```

---

### Task 2: Citation validity and refusal detection (M7)

**Files:**
- Create: `src/main/java/com/learnings/rag/generation/CitationValidator.java`
- Modify: `src/main/java/com/learnings/rag/generation/AnswerService.java`
- Test: `src/test/java/com/learnings/rag/generation/CitationValidatorTest.java` (new); `AnswerServiceTest` (modify)

**Interfaces:**
- Produces:
  - `CitationValidator.check(String answer, int sourceCount)` returns `CitationValidator.Check(List<Integer> cited, List<Integer> outOfRange)`, with `boolean valid()`;
  - `AnswerService.PROMPT_REFUSAL` (`"I couldn't find this in the indexed documentation."`);
  - `static boolean AnswerService.isRefusal(String answer)`.

- [ ] **Step 1: Write the failing tests**

Create `src/test/java/com/learnings/rag/generation/CitationValidatorTest.java`:

```java
package com.learnings.rag.generation;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class CitationValidatorTest {

    @Test
    void citationsOfExistingSourcesAreValid() {
        CitationValidator.Check check = CitationValidator.check("HNSW is the default [1]. IVFFlat builds faster [2][3].", 3);

        assertThat(check.cited()).containsExactly(1, 2, 3);
        assertThat(check.outOfRange()).isEmpty();
        assertThat(check.valid()).isTrue();
    }

    @Test
    void aCitationWithNoMatchingSourceIsInvalid() {
        CitationValidator.Check check = CitationValidator.check("HNSW [1], IVFFlat [7], none [0].", 5);

        assertThat(check.outOfRange()).containsExactly(7, 0);
        assertThat(check.valid()).isFalse();
    }

    @Test
    void anAnswerThatCitesNothingIsInvalid() {
        assertThat(CitationValidator.check("HNSW is the default index type.", 3).valid()).isFalse();
    }

    @Test
    void numbersInCodeAreNotCitations() {
        String answer = "Read `parts[1]` and see [2].\n```java\nint first = values[0];\n```\nAlso [1].";

        assertThat(CitationValidator.check(answer, 2).cited()).containsExactly(2, 1);
    }

    @Test
    void anUnterminatedCodeFenceHidesTheRestOfTheAnswer() {
        assertThat(CitationValidator.check("Use it [1].\n```yaml\nlist: [9]", 1).cited()).containsExactly(1);
    }

    @Test
    void aRepeatedCitationCountsOnce() {
        assertThat(CitationValidator.check("[2] first, [2] again, then [1].", 2).cited()).containsExactly(2, 1);
    }
}
```

Add to `src/test/java/com/learnings/rag/generation/AnswerServiceTest.java`:
1. The imports `java.io.IOException` and `static java.nio.charset.StandardCharsets.UTF_8`.
2. These tests:

```java
    @Test
    void refusalsAreRecognisedHoweverTheModelPhrasesThem() {
        assertThat(AnswerService.isRefusal(AnswerService.NO_SOURCES_ANSWER)).isTrue();
        assertThat(AnswerService.isRefusal(AnswerService.PROMPT_REFUSAL)).isTrue();
        assertThat(AnswerService.isRefusal("Direct answer: I couldn’t find this in the indexed documentation.")).isTrue();
        assertThat(AnswerService.isRefusal("I COULDN'T FIND THIS IN THE INDEXED DOCUMENTATION")).isTrue();
        assertThat(AnswerService.isRefusal("HNSW is the default index type [1].")).isFalse();
        assertThat(AnswerService.isRefusal("If you couldn't find the property, set index-type [1].")).isFalse();
    }

    @Test
    void theSystemPromptStillUsesTheRefusalSentence() throws IOException {
        assertThat(new ClassPathResource("prompts/answer-system.st").getContentAsString(UTF_8))
                .contains(AnswerService.PROMPT_REFUSAL);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./mvnw -q test -Dtest='CitationValidatorTest,AnswerServiceTest'`
Expected: COMPILATION ERRORs: `CitationValidator`, `isRefusal` and `PROMPT_REFUSAL` don't exist.

- [ ] **Step 3: Write the validator**

Create `src/main/java/com/learnings/rag/generation/CitationValidator.java`:

```java
package com.learnings.rag.generation;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Checks an answer's {@code [n]} citations against the number of sources it was given. Code is skipped (fenced
 * blocks, including an unterminated one, and inline code), where {@code parts[1]} is an index, not a citation; the
 * UI's chips don't make that distinction.
 */
public final class CitationValidator {

    private static final Pattern FENCED_CODE = Pattern.compile("(?s)```.*?(?:```|$)");
    private static final Pattern INLINE_CODE = Pattern.compile("`[^`\\n]*`");
    private static final Pattern CITATION = Pattern.compile("\\[(\\d{1,4})]");

    private CitationValidator() {
    }

    /**
     * @param cited the distinct source numbers cited, in order of first appearance
     * @param outOfRange the cited numbers with no matching source
     */
    public record Check(List<Integer> cited, List<Integer> outOfRange) {

        /** At least one citation, and every one points at a source. */
        public boolean valid() {
            return !cited.isEmpty() && outOfRange.isEmpty();
        }
    }

    public static Check check(String answer, int sourceCount) {
        String prose = INLINE_CODE.matcher(FENCED_CODE.matcher(answer).replaceAll(" ")).replaceAll(" ");
        Set<Integer> cited = new LinkedHashSet<>();
        Matcher citation = CITATION.matcher(prose);
        while (citation.find()) {
            cited.add(Integer.parseInt(citation.group(1)));
        }
        List<Integer> outOfRange = cited.stream().filter(n -> n < 1 || n > sourceCount).toList();
        return new Check(List.copyOf(cited), outOfRange);
    }
}
```

- [ ] **Step 4: Recognise refusals**

In `src/main/java/com/learnings/rag/generation/AnswerService.java`:
1. Add the import `java.util.Locale`.
2. After the `ANSWER_FAILED` constant, add:

```java
    /** The refusal sentence the system prompt asks for (prompts/answer-system.st). */
    public static final String PROMPT_REFUSAL = "I couldn't find this in the indexed documentation.";

    private static final List<String> REFUSALS = List.of(
            "i couldn't find this in the indexed documentation",
            "i couldn't find anything about that in the indexed documentation");

    /**
     * Whether an answer is a refusal: the no-sources answer, or the prompt's refusal sentence anywhere in it (models
     * sometimes prefix it, e.g. "Direct answer: I couldn't find this…"). Case and curly apostrophes don't matter.
     */
    public static boolean isRefusal(String answer) {
        String normalized = answer.replace('’', '\'').toLowerCase(Locale.ROOT);
        return REFUSALS.stream().anyMatch(normalized::contains);
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest='CitationValidatorTest,AnswerServiceTest'`
Expected: PASS.

- [ ] **Step 6: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add src/main/java/com/learnings/rag/generation src/test/java/com/learnings/rag/generation
git commit -m "feat: citation validator and refusal detection for answers"
```

---

### Task 3: `POST /api/retrieve` (M7)

**Files:**
- Create: `src/main/java/com/learnings/rag/retrieval/RetrieveRequest.java`, `RetrieveResponse.java`, `RetrievalController.java`
- Test: `src/test/java/com/learnings/rag/retrieval/RetrieveRequestTest.java`, `RetrievalControllerIT.java` (new)
- Modify: `README.md` (HTTP API)

**Interfaces:**
- Consumes: `RetrievalPipeline.retrieve(String, RetrievalOptions)`, `RetrievalOptions.from/with*`, `PipelineTrace` (M2–M6).
- Produces:
  - `POST /api/retrieve`, which answers with `RetrieveResponse(List<RetrieveResponse.Chunk> chunks, PipelineTrace trace)`;
  - `RetrieveResponse.Chunk(int rank, String id, String sourcePath, String breadcrumb, Double score, String text)`.

- [ ] **Step 1: Write the failing tests**

Create `src/test/java/com/learnings/rag/retrieval/RetrieveRequestTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class RetrieveRequestTest {

    private static final RetrievalOptions DEFAULTS = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, false, 0,
            true, 6);

    @Test
    void missingSettingsKeepTheDefaults() {
        assertThat(new RetrieveRequest("q", null, null, null, null, null, null).applyTo(DEFAULTS)).isEqualTo(DEFAULTS);
    }

    @Test
    void givenSettingsOverrideTheDefaults() {
        assertThat(new RetrieveRequest("q", RetrievalMode.VECTOR, 10, true, 3, false, 2.5).applyTo(DEFAULTS))
                .isEqualTo(new RetrievalOptions(10, 0.0, RetrievalMode.VECTOR, 20, true, 3, false, 2.5));
    }
}
```

Create `src/test/java/com/learnings/rag/retrieval/RetrievalControllerIT.java`:

```java
package com.learnings.rag.retrieval;

import static org.hamcrest.Matchers.contains;
import static org.hamcrest.Matchers.lessThanOrEqualTo;
import static org.springframework.http.MediaType.APPLICATION_JSON;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.simple.JdbcClient;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.test.web.servlet.ResultActions;

import com.learnings.rag.RagIntegrationTest;
import com.learnings.rag.ingest.CorpusIngestor;

@RagIntegrationTest
class RetrievalControllerIT {

    @Autowired
    MockMvc mvc;

    @Autowired
    CorpusIngestor corpusIngestor;

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

    private ResultActions retrieve(String json) throws Exception {
        return mvc.perform(post("/api/retrieve").contentType(APPLICATION_JSON).content(json));
    }

    @Test
    void returnsTheChunksAndTheTraceOfTheRequestedMode() throws Exception {
        retrieve("{\"question\":\"spring.ai.vectorstore.pgvector.index-type\",\"mode\":\"KEYWORD\",\"rerank\":false}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks[0].rank").value(1))
                .andExpect(jsonPath("$.chunks[0].sourcePath").value("pgvector.adoc"))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("keyword")));
        retrieve("{\"question\":\"Which index type is HNSW?\",\"mode\":\"VECTOR\",\"rerank\":false,\"topK\":2}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks.length()").value(2))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("vector")));
    }

    @Test
    void aQuestionAloneUsesTheChatDefaults() throws Exception {
        // Hybrid, top 5, rerank on. The test context's stub chat model can't rate, so reranking falls back.
        retrieve("{\"question\":\"Which index type is HNSW and how does it build its graph?\"}")
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.chunks.length()").value(lessThanOrEqualTo(5)))
                .andExpect(jsonPath("$.trace.stages[*].name").value(contains("vector", "keyword", "fusion",
                        PipelineTrace.RERANK_FAILED)));
    }

    @ParameterizedTest
    @ValueSource(strings = {
            "{\"question\":\"   \"}",
            "{\"question\":\"q\",\"topK\":0}",
            "{\"question\":\"q\",\"topK\":21}",
            "{\"question\":\"q\",\"queryVariants\":6}",
            "{\"question\":\"q\",\"minScore\":10.5}",
            "{\"question\":\"q\",\"mode\":\"BM25\"}" })
    void outOfRangeOrUnknownSettingsAreRejected(String body) throws Exception {
        retrieve(body).andExpect(status().isBadRequest());
    }
}
```

- [ ] **Step 2: Run the unit test to verify it fails**

Run: `./mvnw -q test -Dtest=RetrieveRequestTest`
Expected: COMPILATION ERROR, "cannot find symbol: class RetrieveRequest".

- [ ] **Step 3: Write the request, the response and the controller**

Create `src/main/java/com/learnings/rag/retrieval/RetrieveRequest.java`:

```java
package com.learnings.rag.retrieval;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * A retrieval-only request. Every setting but the question is optional and falls back to its {@code rag.retrieval.*}
 * default; the bounds keep a debugging request from asking for unbounded work.
 */
public record RetrieveRequest(@NotBlank @Size(max = 2000) String question,
        RetrievalMode mode,
        @Min(1) @Max(20) Integer topK,
        Boolean rewrite,
        @Min(0) @Max(5) Integer queryVariants,
        Boolean rerank,
        @DecimalMin("0") @DecimalMax("10") Double minScore) {

    RetrievalOptions applyTo(RetrievalOptions defaults) {
        RetrievalOptions options = defaults;
        if (mode != null) {
            options = options.withMode(mode);
        }
        if (topK != null) {
            options = options.withTopK(topK);
        }
        if (rewrite != null) {
            options = options.withRewrite(rewrite);
        }
        if (queryVariants != null) {
            options = options.withQueryVariants(queryVariants);
        }
        if (rerank != null) {
            options = options.withRerank(rerank);
        }
        if (minScore != null) {
            options = options.withMinScore(minScore);
        }
        return options;
    }
}
```

Create `src/main/java/com/learnings/rag/retrieval/RetrieveResponse.java`:

```java
package com.learnings.rag.retrieval;

import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

import org.springframework.ai.document.Document;

import com.learnings.rag.ingest.ChunkMetadata;

/**
 * @param chunks the retrieved chunks, best first, as the chat would hand them to the model
 * @param trace every stage with its queries, ranked hits, scores and timing
 */
public record RetrieveResponse(List<Chunk> chunks, PipelineTrace trace) {

    /** @param score what the chunk was ranked by in the last stage (cosine, ts_rank, RRF or the 0–10 rating) */
    public record Chunk(int rank, String id, String sourcePath, String breadcrumb, Double score, String text) {
    }

    static RetrieveResponse of(RetrievalResult result) {
        List<Document> documents = result.documents();
        return new RetrieveResponse(IntStream.range(0, documents.size())
                .mapToObj(i -> {
                    Document document = documents.get(i);
                    return new Chunk(i + 1, document.getId(),
                            Objects.toString(document.getMetadata().get(ChunkMetadata.SOURCE_PATH), ""),
                            Objects.toString(document.getMetadata().get(ChunkMetadata.BREADCRUMB), ""),
                            document.getScore(), document.getText());
                })
                .toList(), result.trace());
    }
}
```

Create `src/main/java/com/learnings/rag/retrieval/RetrievalController.java`:

```java
package com.learnings.rag.retrieval;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import com.learnings.rag.config.RagProperties;

import jakarta.validation.Valid;

/** Retrieval only, never an answer: the chunks and the full trace, for debugging and comparing settings. */
@RestController
public class RetrievalController {

    private final RetrievalPipeline pipeline;
    private final RetrievalOptions defaults;

    public RetrievalController(RetrievalPipeline pipeline, RagProperties properties) {
        this.pipeline = pipeline;
        this.defaults = RetrievalOptions.from(properties);
    }

    @PostMapping("/api/retrieve")
    public RetrieveResponse retrieve(@Valid @RequestBody RetrieveRequest request) {
        return RetrieveResponse.of(pipeline.retrieve(request.question().strip(), request.applyTo(defaults)));
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest=RetrieveRequestTest`
Expected: PASS, 2 tests.

- [ ] **Step 5: Document the endpoint**

In `README.md`'s `## HTTP API` table, add this row after the `/api/chat` row:

```markdown
| `POST` | `/api/retrieve` | Retrieval only: the chunks and the full trace, no answer | `200` JSON | `400` blank question, out-of-range setting or unknown mode |
```

Add this section after the `### POST /api/chat` section (directly before `### POST /api/ingest/corpus`):

````markdown
### `POST /api/retrieve`

Runs retrieval only and returns the chunks the chat would see, plus every stage of the trace. It never calls the answer
model. Every setting but `question` is optional and defaults to its `rag.retrieval.*` value:

```bash
curl -s -X POST localhost:8081/api/retrieve -H 'Content-Type: application/json' \
     -d '{"question":"spring.ai.vectorstore.pgvector.index-type","mode":"VECTOR","rerank":false}'
```

| Field | Values |
|---|---|
| `mode` | `VECTOR`, `KEYWORD` or `HYBRID` |
| `topK` | 1–20 |
| `rewrite`, `rerank` | `true` / `false` |
| `queryVariants` | 0–5 |
| `minScore` | 0–10 (only with `rerank`) |

The response is `{"chunks": [{rank, id, sourcePath, breadcrumb, score, text}], "trace": {"stages": [{name,
elapsedMillis, hits: [{id, sourcePath, breadcrumb, score}], queries}], "totalMillis"}}`.
````

- [ ] **Step 6: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS, including `RetrievalControllerIT` (3 tests plus 6 parameterized cases).

```bash
git add src/main/java/com/learnings/rag/retrieval src/test/java/com/learnings/rag/retrieval README.md
git commit -m "feat: POST /api/retrieve returns the chunks and the full trace without calling the answer model"
```

---

### Task 4: Debug trace in chat and the UI panel (M7)

**Files:**
- Modify: `src/main/java/com/learnings/rag/generation/ChatRequest.java`, `ChatEvent.java`, `AnswerService.java`, `ChatController.java`
- Modify: `src/main/resources/static/index.html`, `app.js`, `app.css`
- Test: `AnswerServiceTest`, `ChatControllerTest`
- Modify: `README.md` (the SSE event table)

**Interfaces:**
- Consumes: `PipelineTrace` (M2–M6).
- Produces:
  - `ChatRequest(String question, boolean debug)`;
  - `ChatEvent.Trace(PipelineTrace trace)`;
  - `Flux<ChatEvent> AnswerService.answer(String question, boolean debug)`. `answer(String)` is unchanged and means `debug == false`;
  - the SSE event name `trace`.

- [ ] **Step 1: Write the failing tests**

In `src/test/java/com/learnings/rag/generation/AnswerServiceTest.java`, add:

```java
    @Test
    void debugStreamsTheTraceRightAfterTheSources() {
        PipelineTrace trace = new PipelineTrace(List.of(new PipelineTrace.Stage("vector", 3, List.of())), 3);
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(oneChunk().documents(), trace));

        StepVerifier.create(service(new StubChatModel("ok [1]")).answer("q", true))
                .expectNextMatches(ChatEvent.Sources.class::isInstance)
                .expectNext(new ChatEvent.Trace(trace))
                .expectNext(new ChatEvent.Token("ok [1]"))
                .expectNextMatches(ChatEvent.Done.class::isInstance)
                .verifyComplete();
    }

    @Test
    void debugStreamsTheTraceEvenWhenNothingWasFound() {
        PipelineTrace trace = new PipelineTrace(List.of(new PipelineTrace.Stage("rerank", 900, List.of())), 900);
        when(pipeline.retrieve(anyString())).thenReturn(new RetrievalResult(List.of(), trace));

        StepVerifier.create(service(new StubChatModel("never")).answer("sourdough?", true))
                .expectNext(new ChatEvent.Sources(List.of()), new ChatEvent.Trace(trace),
                        new ChatEvent.Token(AnswerService.NO_SOURCES_ANSWER))
                .expectNextMatches(ChatEvent.Done.class::isInstance)
                .verifyComplete();
    }
```

In `src/test/java/com/learnings/rag/generation/ChatControllerTest.java`:
1. In `streamsEventsAsNamedServerSentEvents`, change `when(answerService.answer("What is HNSW?"))` to `when(answerService.answer("What is HNSW?", false))`.
2. Add the import `com.learnings.rag.retrieval.PipelineTrace` and this test:

```java
    @Test
    void aDebugRequestAlsoStreamsTheTrace() throws Exception {
        PipelineTrace trace = new PipelineTrace(List.of(new PipelineTrace.Stage("vector", 3, List.of())), 3);
        when(answerService.answer("What is HNSW?", true)).thenReturn(Flux.just(
                new ChatEvent.Sources(List.of()), new ChatEvent.Trace(trace), new ChatEvent.Token("t"),
                new ChatEvent.Done(null, null, 3, 0)));

        MvcResult started = mvc.perform(post("/api/chat").contentType(APPLICATION_JSON).accept(TEXT_EVENT_STREAM)
                        .content("{\"question\":\"What is HNSW?\",\"debug\":true}"))
                .andExpect(request().asyncStarted())
                .andReturn();

        assertThat(mvc.perform(asyncDispatch(started)).andReturn().getResponse().getContentAsString())
                .containsSubsequence("event:sources", "event:trace", "\"name\":\"vector\"", "\"totalMillis\":3",
                        "event:token");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./mvnw -q test -Dtest='AnswerServiceTest,ChatControllerTest'`
Expected: COMPILATION ERRORs: `ChatEvent.Trace` and `answer(String, boolean)` don't exist.

- [ ] **Step 3: Add the debug flag and the trace event**

Replace `src/main/java/com/learnings/rag/generation/ChatRequest.java` with:

```java
package com.learnings.rag.generation;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/** @param debug also stream the retrieval trace (a {@code trace} event right after {@code sources}) */
public record ChatRequest(@NotBlank @Size(max = 2000) String question, boolean debug) {
}
```

In `src/main/java/com/learnings/rag/generation/ChatEvent.java`:
1. Change the interface javadoc to `/** The SSE protocol of /api/chat: one Sources, a Trace when debugging, any number of Tokens, then Done, or Error at any point. */`.
2. Add the import `com.learnings.rag.retrieval.PipelineTrace`.
3. After the `Sources` record, add:

```java
    /** Every retrieval stage with its hits and timing; sent only when the request asks for debugging. */
    record Trace(PipelineTrace trace) implements ChatEvent {
    }
```

In `src/main/java/com/learnings/rag/generation/AnswerService.java`:
1. Replace the method `public Flux<ChatEvent> answer(String question) {` and its body with:

```java
    public Flux<ChatEvent> answer(String question) {
        return answer(question, false);
    }

    /** @param debug also stream the retrieval trace, right after the sources */
    public Flux<ChatEvent> answer(String question, boolean debug) {
        return Mono.fromCallable(() -> retrievalPipeline.retrieve(question))
                .subscribeOn(Schedulers.boundedElastic()) // JDBC + embedding call are blocking
                .flatMapMany(retrieval -> generate(question, retrieval, debug))
                .onErrorResume(error -> {
                    // Details (SQL errors, API error bodies) stay in the log; the browser gets a generic message.
                    log.error("Answering failed for question: {}", question, error);
                    return Flux.just(new ChatEvent.Error(ANSWER_FAILED));
                });
    }
```

2. In `generate`, change the signature to `private Flux<ChatEvent> generate(String question, RetrievalResult retrieval, boolean debug) {`. After `long retrievalMillis = retrieval.trace().totalMillis();`, add:

```java
        Flux<ChatEvent> head = debug ? Flux.just(sources, new ChatEvent.Trace(retrieval.trace())) : Flux.just(sources);
```

3. Replace `return Flux.just(sources, new ChatEvent.Token(NO_SOURCES_ANSWER),\n                    new ChatEvent.Done(null, null, retrievalMillis, 0));` with:

```java
            return Flux.concat(head, Flux.just(new ChatEvent.Token(NO_SOURCES_ANSWER),
                    new ChatEvent.Done(null, null, retrievalMillis, 0)));
```

4. Replace `return Flux.concat(Mono.just(sources), tokens, done);` with `return Flux.concat(head, tokens, done);`.

In `src/main/java/com/learnings/rag/generation/ChatController.java`:
1. Replace `return answerService.answer(request.question().strip())` with `return answerService.answer(request.question().strip(), request.debug())`.
2. Add `case ChatEvent.Trace _ -> "trace";` to the `eventName` switch after the `Sources` case.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest='AnswerServiceTest,ChatControllerTest'`
Expected: PASS. The existing `streamsSourcesThenTokensThenUsage` still passes: without `debug`, no trace event is sent.

- [ ] **Step 5: Add the Debug toggle and the trace panel**

In `src/main/resources/static/index.html`, replace `      <button type="submit" id="ask-button">Ask</button>` with:

```html
      <div class="ask-row">
        <label class="debug-toggle"><input type="checkbox" id="debug"> Debug trace</label>
        <button type="submit" id="ask-button">Ask</button>
      </div>
```

Then, after `    <ol id="sources" class="sources"></ol>`, add:

```html
    <h2 id="trace-heading" hidden>Retrieval trace</h2>
    <div id="trace" class="trace"></div>
```

In `src/main/resources/static/app.js`:
1. In the submit handler, after `renderSources([]);`, add `renderTrace(null);`.
2. Change `body: JSON.stringify({ question }),` to `body: JSON.stringify({ question, debug: $('#debug').checked }),`.
3. After the `if (event === 'sources') { … }` block, add:

```js
      } else if (event === 'trace') {
        renderTrace(data.trace);
```

so the chain reads `if (event === 'sources') {…} else if (event === 'trace') {…} else if (event === 'token') {…}`.

4. Add these functions after `renderSources`:

```js
// Debug: every retrieval stage with its timing, the queries it searched and its ranked chunks with their scores.
function renderTrace(trace) {
  $('#trace-heading').hidden = !trace;
  if (!trace) {
    $('#trace').replaceChildren();
    return;
  }
  const stages = trace.stages.map((stage) => element('details', {},
    element('summary', { textContent: stageSummary(stage) }),
    ...(stage.queries.length ? [element('ul', { className: 'trace-queries' },
      ...stage.queries.map((q) => element('li', { textContent: q })))] : []),
    element('ol', { className: 'trace-hits' }, ...stage.hits.map((hit) => element('li', {},
      element('span', { textContent: hit.breadcrumb ? `${hit.sourcePath} › ${hit.breadcrumb}` : hit.sourcePath }),
      element('span', { className: 'score', textContent: traceScore(stage.name, hit.score) }))))));
  $('#trace').replaceChildren(
    element('p', { className: 'stats', textContent: `retrieval ${trace.totalMillis} ms (wall clock; stages can overlap)` }),
    ...stages);
}

function stageSummary(stage) {
  if (stage.name === 'rerank-failed') return `rerank failed · ${stage.elapsedMillis} ms · fused order kept`;
  const parts = [stage.name, `${stage.elapsedMillis} ms`];
  if (stage.queries.length) parts.push(`${stage.queries.length} ${stage.queries.length === 1 ? 'query' : 'queries'}`);
  if (stage.hits.length) parts.push(`${stage.hits.length} chunks`);
  return parts.join(' · ');
}

function traceScore(stageName, score) {
  if (score == null) return '';
  return stageName === 'rerank' ? `${Number.isInteger(score) ? score : score.toFixed(1)}/10` : score.toFixed(4);
}
```

In `src/main/resources/static/app.css`, add before the `@media (max-width: 600px)` block:

```css
.ask-row { display: flex; gap: 14px; align-items: center; }
.debug-toggle { color: var(--muted); font-size: 13px; display: flex; gap: 6px; align-items: center; }
.trace details { border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; margin-bottom: 8px;
                 background: var(--surface); }
.trace summary { cursor: pointer; font-size: 14px; }
.trace-hits, .trace-queries { font-size: 13px; margin: 8px 0 4px; padding-left: 22px; }
.trace-hits li { display: flex; justify-content: space-between; gap: 12px; }
```

Run: `node --check src/main/resources/static/app.js`
Expected: no output (the syntax is valid). The browser check is in Task 7.

- [ ] **Step 6: Document the trace event**

In `README.md`'s SSE event table:
1. Replace the `sources` row with:

```markdown
| `sources` | `{"sources": [{n, sourcePath, title, breadcrumb, score, text, scores}]}` | Once, first. `score` is what the chunk was ranked by last: the reranker's 0–10 rating by default, otherwise the fused RRF score (hybrid) or cosine similarity (vector). `scores` gives each retrieval stage's score (`vector`, `keyword`, `fusion`, `rerank`). `n` is the number the model cites. |
```

2. Add this row after it:

```markdown
| `trace` | `{"trace": {"stages": [{name, elapsedMillis, hits, queries}], "totalMillis"}}` | Only when the request has `"debug": true`, right after `sources`. Every retrieval stage with its ranked hits and scores. |
```

3. After the table's following paragraph (the one about `EventSource`), add: `Add "debug": true to the request body to get the trace event; the UI's **Debug trace** toggle does this and shows the trace under the sources.`

- [ ] **Step 7: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add src/main/java/com/learnings/rag/generation src/main/resources/static src/test/java/com/learnings/rag/generation README.md
git commit -m "feat: chat streams the retrieval trace on request, and the UI shows it in a debug panel"
```

---

### Task 5: The answer judge and the generation runner (M7)

**Files:**
- Create: `src/main/java/com/learnings/rag/eval/AnswerJudge.java`, `GenerationReport.java`, `GenerationEvalRunner.java`
- Test: `src/test/java/com/learnings/rag/eval/AnswerJudgeTest.java`, `GenerationEvalRunnerTest.java` (new)

**Interfaces:**
- Consumes:
  - `AnswerService.answer(String)`, `AnswerService.isRefusal`, `AnswerService.NO_SOURCES_ANSWER`, `CitationValidator.check` (Task 2);
  - the `ChatEvent` variants, including `Trace` (Task 4);
  - `GoldenItem.answerable()`, `GoldenItem.referenceAnswer()` (M3/M6);
  - `RetrievalMetrics.percentile` (M3).
- Produces:
  - `AnswerJudge.Verdict {PASS, FAIL, ERROR}`;
  - `AnswerJudge.faithful(String answer, List<String> sources)`, `relevant(String question, String answer, List<String> sources)` and `correct(String answer, String referenceAnswer)`, each returning `Verdict`;
  - `GenerationReport(String answerModel, String judgeModel, RetrievalOptions retrieval, Group answerable, Group unanswerable, long p50GenerationMillis, List<Item> items)`, with `int judgeErrors()`;
  - `GenerationReport.Outcome {ANSWERED, REFUSED_BY_RETRIEVAL, REFUSED_BY_MODEL, FAILED}`;
  - `GenerationReport.Item(String id, String question, boolean answerable, Outcome outcome, int sources, String answer, Verdict faithful, Verdict relevant, Verdict correct, CitationValidator.Check citations, long generationMillis)`;
  - `GenerationReport.Rate(int passed, int judged, int errors)`;
  - `GenerationReport.Group(int questions, int answered, int refusedByRetrieval, int refusedByModel, int failed, Rate faithful, Rate relevant, Rate correct, Rate citationsValid)`, with `static Group of(List<Item>)`;
  - `GenerationReport GenerationEvalRunner.run(List<GoldenItem> items)`.

- [ ] **Step 1: Write the failing tests**

Create `src/test/java/com/learnings/rag/eval/AnswerJudgeTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.eval.AnswerJudge.Verdict;

class AnswerJudgeTest {

    private static final List<String> SOURCES = List.of("PGvector › Indexes\n\nHNSW is the default index type.");

    private static AnswerJudge judge(ChatModel model) {
        return new AnswerJudge(ChatClient.builder(model).build());
    }

    @Test
    void yesInAnyCasePassesAndNoFails() {
        assertThat(judge(new StubChatModel("yes")).faithful("HNSW is the default [1].", SOURCES)).isEqualTo(Verdict.PASS);
        assertThat(judge(new StubChatModel("YES")).relevant("Default index?", "HNSW [1].", SOURCES)).isEqualTo(Verdict.PASS);
        assertThat(judge(new StubChatModel("No")).correct("IVFFlat [1].", "HNSW is the default.")).isEqualTo(Verdict.FAIL);
    }

    @Test
    void aDecoratedYesFails() {
        // Spring AI's evaluators accept exactly "yes" in any case; pinned so that an upgrade changing it is noticed.
        assertThat(judge(new StubChatModel("Yes.")).faithful("HNSW is the default [1].", SOURCES)).isEqualTo(Verdict.FAIL);
    }

    @Test
    void aFailingJudgeIsAnErrorNotAFail() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        assertThat(judge(unavailable).faithful("HNSW [1].", SOURCES)).isEqualTo(Verdict.ERROR);
    }

    @Test
    void faithfulnessChecksTheAnswerAgainstItsSources() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).faithful("HNSW is the default [1].", SOURCES);

        assertThat(model.prompts().getFirst().getContents())
                .containsSubsequence("Document:", "HNSW is the default index type.", "Claim:", "HNSW is the default [1].");
    }

    @Test
    void correctnessChecksTheReferenceAnswerAgainstTheAnswer() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).correct("Set index-type to HNSW [1].", "HNSW is the default.");

        assertThat(model.prompts().getFirst().getContents())
                .containsSubsequence("Document:", "Set index-type to HNSW [1].", "Claim:", "HNSW is the default.");
    }

    @Test
    void relevancySeesTheQuestionTheAnswerAndTheSources() {
        StubChatModel model = new StubChatModel("yes");

        judge(model).relevant("Which index is the default?", "HNSW [1].", SOURCES);

        assertThat(model.prompts().getFirst().getContents()).containsSubsequence("Query:", "Which index is the default?",
                "Response:", "HNSW [1].", "Context:", "HNSW is the default index type.");
    }

    @Test
    void bracesAndCodeReachTheJudgesVerbatim() {
        StubChatModel model = new StubChatModel("yes");
        List<String> sources = List.of("Use {name} or {{double}} and ${user.home} in templates.");

        assertThat(judge(model).faithful("Write `{name}` [1].", sources)).isEqualTo(Verdict.PASS);
        assertThat(judge(model).relevant("What does {name} do?", "Write `{name}` [1].", sources)).isEqualTo(Verdict.PASS);

        assertThat(model.prompts()).allSatisfy(prompt -> assertThat(prompt.getContents())
                .contains("Use {name} or {{double}} and ${user.home} in templates.", "Write `{name}` [1]."));
    }
}
```

Create `src/test/java/com/learnings/rag/eval/GenerationEvalRunnerTest.java`:

```java
package com.learnings.rag.eval;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.AnswerJudge.Verdict;
import com.learnings.rag.eval.GenerationReport.Group;
import com.learnings.rag.eval.GenerationReport.Item;
import com.learnings.rag.eval.GenerationReport.Outcome;
import com.learnings.rag.eval.GenerationReport.Rate;
import com.learnings.rag.generation.AnswerService;
import com.learnings.rag.generation.ChatEvent;
import com.learnings.rag.generation.SourceRef;
import com.learnings.rag.retrieval.RetrievalMode;

import reactor.core.publisher.Flux;

class GenerationEvalRunnerTest {

    private final AnswerService answers = mock(AnswerService.class);
    private final StubChatModel judgeModel = new StubChatModel("yes");

    private GenerationEvalRunner runner(ChatModel judge) {
        RagProperties properties = new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, "gpt-4.1-mini",
                        new RagProperties.Rerank(true, 6)));
        return new GenerationEvalRunner(answers, new AnswerJudge(ChatClient.builder(judge).build()), properties,
                "gpt-5-mini");
    }

    private static GoldenItem answerable(String id) {
        return new GoldenItem(id, id + "?", List.of(new ExpectedSource("a.adoc", "")), "HNSW is the default.", null);
    }

    private static GoldenItem unanswerable(String id) {
        return new GoldenItem(id, id + "?", List.of(), "Not covered.", null, List.of(GoldenItem.UNANSWERABLE));
    }

    private static List<SourceRef> sources(int count) {
        return IntStream.rangeClosed(1, count)
                .mapToObj(n -> new SourceRef(n, "a.adoc", "A", "S" + n, 8.0, "source text " + n))
                .toList();
    }

    private static Flux<ChatEvent> answered(String text, int sourceCount) {
        return Flux.just(new ChatEvent.Sources(sources(sourceCount)), new ChatEvent.Token(text),
                new ChatEvent.Done(100, 20, 5, 900));
    }

    @Test
    void answeredQuestionsAreJudgedAndTheirCitationsChecked() {
        when(answers.answer("q1?")).thenReturn(answered("HNSW is the default [1].", 2));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1")));

        Item item = report.items().getFirst();
        assertThat(item.outcome()).isEqualTo(Outcome.ANSWERED);
        assertThat(List.of(item.faithful(), item.relevant(), item.correct())).containsOnly(Verdict.PASS);
        assertThat(item.citations().valid()).isTrue();
        assertThat(judgeModel.prompts()).hasSize(3);
        assertThat(report.answerable()).isEqualTo(new Group(1, 1, 0, 0, 0, new Rate(1, 1, 0), new Rate(1, 1, 0),
                new Rate(1, 1, 0), new Rate(1, 1, 0)));
        assertThat(report.p50GenerationMillis()).isEqualTo(900);
        assertThat(report.answerModel()).isEqualTo("gpt-5-mini");
        assertThat(report.judgeModel()).isEqualTo("gpt-4.1-mini");
    }

    @Test
    void refusalsAreClassifiedNotJudged() {
        when(answers.answer("q1?")).thenReturn(Flux.just(new ChatEvent.Sources(List.of()),
                new ChatEvent.Token(AnswerService.NO_SOURCES_ANSWER), new ChatEvent.Done(null, null, 5, 0)));
        when(answers.answer("q2?")).thenReturn(answered("Direct answer: I couldn't find this in the indexed documentation.", 3));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1"), answerable("q2")));

        assertThat(report.items()).extracting(Item::outcome)
                .containsExactly(Outcome.REFUSED_BY_RETRIEVAL, Outcome.REFUSED_BY_MODEL);
        assertThat(report.items()).allSatisfy(item -> assertThat(item.faithful()).isNull());
        assertThat(judgeModel.prompts()).isEmpty();
        assertThat(report.answerable()).isEqualTo(new Group(2, 0, 1, 1, 0, new Rate(0, 0, 0), new Rate(0, 0, 0),
                new Rate(0, 0, 0), new Rate(0, 0, 0)));
    }

    @Test
    void aFailedOrEmptyAnswerIsAFailureNotARefusal() {
        when(answers.answer("q1?")).thenReturn(Flux.just(new ChatEvent.Sources(sources(2)), new ChatEvent.Token("Partial "),
                new ChatEvent.Error(AnswerService.ANSWER_FAILED)));
        when(answers.answer("q2?")).thenReturn(Flux.just(new ChatEvent.Sources(sources(2)),
                new ChatEvent.Done(100, 0, 5, 300)));

        GenerationReport report = runner(judgeModel).run(List.of(answerable("q1"), answerable("q2")));

        assertThat(report.items()).extracting(Item::outcome).containsExactly(Outcome.FAILED, Outcome.FAILED);
        assertThat(report.answerable().failed()).isEqualTo(2);
        assertThat(judgeModel.prompts()).isEmpty();
    }

    @Test
    void anAnsweredUnanswerableQuestionIsJudgedButNotForCorrectness() {
        when(answers.answer("u1?")).thenReturn(answered("Use spring.ai.openai.chat.options.model [1].", 2));

        GenerationReport report = runner(judgeModel).run(List.of(unanswerable("u1")));

        assertThat(report.items().getFirst().correct()).isNull();
        assertThat(judgeModel.prompts()).hasSize(2);
        assertThat(report.unanswerable().answered()).isEqualTo(1);
        assertThat(report.unanswerable().correct()).isEqualTo(new Rate(0, 0, 0));
    }

    @Test
    void failedChecksAndJudgeErrorsAreCountedApart() {
        when(answers.answer("q1?")).thenReturn(answered("See [4].", 2));
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        GenerationReport failing = runner(new StubChatModel("no")).run(List.of(answerable("q1")));
        GenerationReport erroring = runner(unavailable).run(List.of(answerable("q1")));

        assertThat(failing.answerable().faithful()).isEqualTo(new Rate(0, 1, 0));
        assertThat(failing.answerable().citationsValid()).isEqualTo(new Rate(0, 1, 0));
        assertThat(erroring.answerable().faithful()).isEqualTo(new Rate(0, 0, 1));
        assertThat(erroring.judgeErrors()).isEqualTo(3);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./mvnw -q test -Dtest='AnswerJudgeTest,GenerationEvalRunnerTest'`
Expected: COMPILATION ERRORs: `AnswerJudge`, `GenerationEvalRunner` and `GenerationReport` don't exist.

- [ ] **Step 3: Write the judge**

Create `src/main/java/com/learnings/rag/eval/AnswerJudge.java`:

```java
package com.learnings.rag.eval;

import java.util.List;
import java.util.function.Supplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.evaluation.FactCheckingEvaluator;
import org.springframework.ai.chat.evaluation.RelevancyEvaluator;
import org.springframework.ai.document.Document;
import org.springframework.ai.evaluation.EvaluationRequest;
import org.springframework.ai.evaluation.EvaluationResponse;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/**
 * LLM-as-judge checks of one answer with Spring AI's evaluators, on the utility model: a different model from the one
 * that wrote the answer, so no model grades itself. Each check is one call that must reply "yes"; Spring AI accepts
 * exactly "yes" in any case, so a decorated reply ("Yes.") is a fail. A failed call is an {@link Verdict#ERROR},
 * never a fail.
 */
@Component
public class AnswerJudge {

    private static final Logger log = LoggerFactory.getLogger(AnswerJudge.class);

    public enum Verdict {
        PASS, FAIL, ERROR
    }

    private final FactCheckingEvaluator factChecker;
    private final RelevancyEvaluator relevancy;

    public AnswerJudge(@Qualifier("utilityChatClient") ChatClient utilityChatClient) {
        this.factChecker = FactCheckingEvaluator.builder(utilityChatClient.mutate()).build();
        this.relevancy = RelevancyEvaluator.builder().chatClientBuilder(utilityChatClient.mutate()).build();
    }

    /** Faithfulness: is the answer supported by the sources it was given? */
    public Verdict faithful(String answer, List<String> sources) {
        return verdict("faithfulness", () -> factChecker.evaluate(new EvaluationRequest(documents(sources), answer)));
    }

    /** Relevancy: does the answer respond to the question, in line with the sources? */
    public Verdict relevant(String question, String answer, List<String> sources) {
        return verdict("relevancy",
                () -> relevancy.evaluate(new EvaluationRequest(question, documents(sources), answer)));
    }

    /** Correctness: is the golden reference answer supported by the answer, i.e. does the answer contain its facts? */
    public Verdict correct(String answer, String referenceAnswer) {
        return verdict("correctness",
                () -> factChecker.evaluate(new EvaluationRequest(List.of(new Document(answer)), referenceAnswer)));
    }

    private static List<Document> documents(List<String> sources) {
        return sources.stream().map(Document::new).toList();
    }

    private static Verdict verdict(String check, Supplier<EvaluationResponse> evaluation) {
        try {
            return evaluation.get().isPass() ? Verdict.PASS : Verdict.FAIL;
        }
        catch (RuntimeException e) {
            log.warn("The {} judge failed; counted as an error, not a fail", check, e);
            return Verdict.ERROR;
        }
    }
}
```

- [ ] **Step 4: Write the report model**

Create `src/main/java/com/learnings/rag/eval/GenerationReport.java`:

```java
package com.learnings.rag.eval;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

import com.learnings.rag.eval.AnswerJudge.Verdict;
import com.learnings.rag.generation.CitationValidator;
import com.learnings.rag.retrieval.RetrievalOptions;

/**
 * What the chat answered for every golden question, and how the answers were judged.
 *
 * @param retrieval the chat's retrieval settings the answers were generated with
 * @param p50GenerationMillis median answer-model time over the answered questions; 0 when none was answered
 */
public record GenerationReport(String answerModel, String judgeModel, RetrievalOptions retrieval, Group answerable,
        Group unanswerable, long p50GenerationMillis, List<Item> items) {

    /** How the chat handled a question. Only answered questions are judged: a judge marks any refusal unsupported. */
    public enum Outcome {
        ANSWERED, REFUSED_BY_RETRIEVAL, REFUSED_BY_MODEL, FAILED
    }

    /**
     * @param sources how many sources the answer model was given
     * @param faithful judge verdicts; null when not judged (not answered, or no reference for correctness)
     * @param citations null when not answered
     */
    public record Item(String id, String question, boolean answerable, Outcome outcome, int sources, String answer,
            Verdict faithful, Verdict relevant, Verdict correct, CitationValidator.Check citations,
            long generationMillis) {
    }

    /** @param judged verdicts that passed or failed; {@code errors} are judge failures, counted apart */
    public record Rate(int passed, int judged, int errors) {

        static Rate of(List<Item> answered, Function<Item, Verdict> verdict) {
            List<Verdict> verdicts = answered.stream().map(verdict).filter(Objects::nonNull).toList();
            int passed = (int) verdicts.stream().filter(v -> v == Verdict.PASS).count();
            int errors = (int) verdicts.stream().filter(v -> v == Verdict.ERROR).count();
            return new Rate(passed, verdicts.size() - errors, errors);
        }
    }

    public record Group(int questions, int answered, int refusedByRetrieval, int refusedByModel, int failed,
            Rate faithful, Rate relevant, Rate correct, Rate citationsValid) {

        static Group of(List<Item> items) {
            List<Item> answered = items.stream().filter(item -> item.outcome() == Outcome.ANSWERED).toList();
            int citationsValid = (int) answered.stream().filter(item -> item.citations().valid()).count();
            return new Group(items.size(), answered.size(), count(items, Outcome.REFUSED_BY_RETRIEVAL),
                    count(items, Outcome.REFUSED_BY_MODEL), count(items, Outcome.FAILED),
                    Rate.of(answered, Item::faithful), Rate.of(answered, Item::relevant),
                    Rate.of(answered, Item::correct), new Rate(citationsValid, answered.size(), 0));
        }

        private static int count(List<Item> items, Outcome outcome) {
            return (int) items.stream().filter(item -> item.outcome() == outcome).count();
        }
    }

    /** Judge calls that failed, over every check and both groups. */
    public int judgeErrors() {
        return List.of(answerable, unanswerable).stream()
                .mapToInt(group -> group.faithful().errors() + group.relevant().errors() + group.correct().errors())
                .sum();
    }
}
```

- [ ] **Step 5: Write the runner**

Create `src/main/java/com/learnings/rag/eval/GenerationEvalRunner.java`:

```java
package com.learnings.rag.eval;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.eval.GenerationReport.Group;
import com.learnings.rag.eval.GenerationReport.Item;
import com.learnings.rag.eval.GenerationReport.Outcome;
import com.learnings.rag.generation.AnswerService;
import com.learnings.rag.generation.ChatEvent;
import com.learnings.rag.generation.CitationValidator;
import com.learnings.rag.generation.SourceRef;
import com.learnings.rag.retrieval.RetrievalOptions;

import reactor.core.publisher.Flux;

/**
 * Sends every golden question through the chat path ({@link AnswerService}, chat defaults) and judges what comes back:
 * refusals are classified, answered questions get faithfulness, relevancy, correctness (answerable only) and a
 * citation check. One question at a time, in golden-set order.
 */
@Service
public class GenerationEvalRunner {

    private static final Logger log = LoggerFactory.getLogger(GenerationEvalRunner.class);

    private static final Duration ANSWER_TIMEOUT = Duration.ofMinutes(5);

    private final AnswerService answerService;
    private final AnswerJudge judge;
    private final RetrievalOptions retrieval;
    private final String answerModel;
    private final String judgeModel;

    public GenerationEvalRunner(AnswerService answerService, AnswerJudge judge, RagProperties properties,
            @Value("${spring.ai.openai.chat.options.model:unknown}") String answerModel) {
        this.answerService = answerService;
        this.judge = judge;
        this.retrieval = RetrievalOptions.from(properties);
        this.answerModel = answerModel;
        this.judgeModel = properties.retrieval().utilityModel();
    }

    public GenerationReport run(List<GoldenItem> items) {
        List<Item> results = new ArrayList<>(items.size());
        for (int i = 0; i < items.size(); i++) {
            Item result = evaluate(items.get(i));
            results.add(result);
            log.info("Generation {} ({}/{}): {} · faithful {} · relevant {} · correct {} · citations {}", result.id(),
                    i + 1, items.size(), result.outcome(), result.faithful(), result.relevant(), result.correct(),
                    result.citations() == null ? null : result.citations().valid() ? "valid" : "invalid");
        }
        List<Long> generation = results.stream().filter(item -> item.outcome() == Outcome.ANSWERED)
                .map(Item::generationMillis).toList();
        return new GenerationReport(answerModel, judgeModel, retrieval,
                Group.of(results.stream().filter(Item::answerable).toList()),
                Group.of(results.stream().filter(item -> !item.answerable()).toList()),
                generation.isEmpty() ? 0 : RetrievalMetrics.percentile(generation, 0.50), List.copyOf(results));
    }

    private Item evaluate(GoldenItem item) {
        Answer answer = collect(item.question());
        Outcome outcome = answer.failed() || answer.text().isBlank() ? Outcome.FAILED
                : answer.sources().isEmpty() ? Outcome.REFUSED_BY_RETRIEVAL
                : AnswerService.isRefusal(answer.text()) ? Outcome.REFUSED_BY_MODEL
                : Outcome.ANSWERED;
        if (outcome != Outcome.ANSWERED) {
            return new Item(item.id(), item.question(), item.answerable(), outcome, answer.sources().size(),
                    answer.text(), null, null, null, null, answer.generationMillis());
        }
        List<String> sources = answer.sources().stream().map(SourceRef::text).toList();
        return new Item(item.id(), item.question(), item.answerable(), outcome, sources.size(), answer.text(),
                judge.faithful(answer.text(), sources),
                judge.relevant(item.question(), answer.text(), sources),
                item.answerable() ? judge.correct(answer.text(), item.referenceAnswer()) : null,
                CitationValidator.check(answer.text(), sources.size()), answer.generationMillis());
    }

    private record Answer(List<SourceRef> sources, String text, boolean failed, long generationMillis) {
    }

    /** The chat's events for one question, folded into one answer; a timeout or an exception is a failed answer. */
    private Answer collect(String question) {
        List<ChatEvent> events;
        try {
            Flux<ChatEvent> stream = answerService.answer(question);
            events = stream.collectList().block(ANSWER_TIMEOUT);
        }
        catch (RuntimeException e) {
            log.warn("Answering failed for question: {}", question, e);
            return new Answer(List.of(), "", true, 0);
        }
        List<SourceRef> sources = List.of();
        StringBuilder text = new StringBuilder();
        boolean failed = false;
        long generationMillis = 0;
        for (ChatEvent event : events == null ? List.<ChatEvent>of() : events) {
            switch (event) {
                case ChatEvent.Sources s -> sources = s.sources();
                case ChatEvent.Trace _ -> {
                }
                case ChatEvent.Token t -> text.append(t.text());
                case ChatEvent.Done d -> generationMillis = d.generationMillis();
                case ChatEvent.Error _ -> failed = true;
            }
        }
        return new Answer(sources, text.toString(), failed, generationMillis);
    }
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest='AnswerJudgeTest,GenerationEvalRunnerTest'`
Expected: PASS, 7 + 5 tests. WARN logs from the failing-judge tests are expected.

If `bracesAndCodeReachTheJudgesVerbatim` fails with a template error, Spring AI's evaluator renders a value as a template. That is a real finding, not a test problem. Rule on it in the ledger: pass the text through our own `Evaluator` built on Message objects, as `AnswerService` does.

- [ ] **Step 7: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS. `RagApplicationIT` proves the `AnswerJudge` and `GenerationEvalRunner` beans wire up.

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval
git commit -m "feat: answer judge on Spring AI's evaluators and a generation runner over the golden set"
```

---

### Task 6: The Generation section of the eval report (M7)

**Files:**
- Modify: `src/main/java/com/learnings/rag/eval/EvalProperties.java`, `EvalReport.java`, `EvalCommand.java`, `ReportWriter.java`
- Test: `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`
- Modify: `eval/README.md` (how to run, metrics)

**Interfaces:**
- Consumes: `GenerationEvalRunner.run`, `GenerationReport` and its records (Task 5); `AnswerService.NO_SOURCES_ANSWER` and `CitationValidator.check` (Task 2).
- Produces:
  - `EvalProperties.generation()` (`rag.eval.generation`, default false);
  - the `EvalReport` component `GenerationReport generation` (4th, nullable), the 3-argument constructor without it, and `EvalReport withGeneration(GenerationReport)`;
  - the report section `## Generation`.

- [ ] **Step 1: Write the failing tests**

In `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`, add the imports `com.learnings.rag.eval.AnswerJudge.Verdict`, `com.learnings.rag.eval.GenerationReport.Group`, `com.learnings.rag.eval.GenerationReport.Item`, `com.learnings.rag.eval.GenerationReport.Outcome`, `com.learnings.rag.generation.AnswerService` and `com.learnings.rag.generation.CitationValidator`. Then add:

```java
    private static GenerationReport generation() {
        Item ok = new Item("q01", "Which index?", true, Outcome.ANSWERED, 2, "HNSW [1].", Verdict.PASS, Verdict.PASS,
                Verdict.PASS, CitationValidator.check("HNSW [1].", 2), 900);
        Item flawed = new Item("q02", "How do I stream?", true, Outcome.ANSWERED, 3, "Use stream() [4].", Verdict.FAIL,
                Verdict.PASS, Verdict.ERROR, CitationValidator.check("Use stream() [4].", 3), 1100);
        Item refused = new Item("u01", "Sourdough?", false, Outcome.REFUSED_BY_RETRIEVAL, 0,
                AnswerService.NO_SOURCES_ANSWER, null, null, null, null, 0);
        RetrievalOptions chat = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20).withRerank(true).withMinScore(6);
        return new GenerationReport("gpt-5-mini", "gpt-4.1-mini", chat, Group.of(List.of(ok, flawed)),
                Group.of(List.of(refused)), 900, List.of(ok, flawed, refused));
    }

    @Test
    void theGenerationSectionSummarizesTheAnswersAndListsTheOnesToReview() {
        String markdown = ReportWriter.markdown(report(0).withGeneration(generation()));

        assertThat(markdown).contains(
                "## Generation",
                "Answer model `gpt-5-mini`, judge `gpt-4.1-mini`",
                "hybrid retrieval, top 5, rerank on, min-score 6",
                "| answerable | 2 | 2 | 0 | 0 | 0 | 1/2 | 2/2 | 1/1 | 1/2 |",
                "| unanswerable | 1 | 0 | 1 | 0 | 0 | – | – | – | – |",
                "p50 generation: 900 ms. Judge errors (not counted as fails): 1.",
                "| q02 | answered | 3 | ✗ | ✓ | error | ✗ | How do I stream? |",
                "| u01 | refused by retrieval | 0 | – | – | – | – | Sourdough? |",
                "- **q02** How do I stream? — faithfulness failed; correctness judge error; citations out of range: 4",
                "  > Use stream() [4].");
        assertThat(markdown).doesNotContain("- **q01**", "- **u01**");
    }

    @Test
    void thereIsNoGenerationSectionWithoutTheFlag() {
        assertThat(ReportWriter.markdown(report(0))).doesNotContain("## Generation");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./mvnw -q test -Dtest=ReportWriterTest`
Expected: COMPILATION ERROR: `withGeneration` doesn't exist.

- [ ] **Step 3: Add the flag and the report component**

In `src/main/java/com/learnings/rag/eval/EvalProperties.java`:
1. Add `@param generation also generate and judge an answer for every question (calls the answer and judge models)` to the javadoc.
2. Change `@DefaultValue Golden golden) {` to `@DefaultValue Golden golden,\n        @DefaultValue("false") boolean generation) {`.

In `src/main/java/com/learnings/rag/eval/EvalReport.java`, replace the line `public record EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs) {` with:

```java
public record EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs, GenerationReport generation) {

    /** A retrieval-only report. */
    public EvalReport(Instant startedAt, RunInfo run, List<ConfigResult> configs) {
        this(startedAt, run, configs, null);
    }

    public EvalReport withGeneration(GenerationReport generation) {
        return new EvalReport(startedAt, run, configs, generation);
    }
```

In `src/main/java/com/learnings/rag/eval/EvalCommand.java`:
1. Add the field `private final GenerationEvalRunner generationRunner;`, the constructor parameter `GenerationEvalRunner generationRunner` after `EvalRunner runner`, and its assignment.
2. Change the class javadoc to `/** {@code ./mvnw spring-boot:run -Dspring-boot.run.profiles=eval}: scores retrieval (and with {@code --rag.eval.generation=true} the answers) against the golden set. */`.
3. Replace the `run` method's body with:

```java
        GoldenSet goldenSet = files.read(properties.goldenSet());
        log.info("Evaluating {} questions from {}", goldenSet.items().size(), goldenSet.path());
        EvalReport report = runner.run(goldenSet, EvalConfig.all(ragProperties)); // logs one summary line per config
        if (properties.generation()) {
            log.info("Generating and judging an answer for each of the {} questions", goldenSet.items().size());
            report = report.withGeneration(generationRunner.run(goldenSet.items()));
        }
        Path markdown = reportWriter.write(report, properties.reportsDir());
        log.info("Report written to {}", markdown);
```

- [ ] **Step 4: Write the Generation section**

In `src/main/java/com/learnings/rag/eval/ReportWriter.java`:
1. Add the imports `java.util.ArrayList`, `com.learnings.rag.eval.AnswerJudge.Verdict`, `com.learnings.rag.eval.GenerationReport.Group`, `com.learnings.rag.eval.GenerationReport.Item`, `com.learnings.rag.eval.GenerationReport.Outcome`, `com.learnings.rag.eval.GenerationReport.Rate` and `com.learnings.rag.retrieval.RetrievalOptions`.
2. In `markdown(...)`, replace the final `return md.toString();` with:

```java
        if (report.generation() != null) {
            generation(md, report.generation());
        }
        return md.toString();
```

3. Add these methods before `decimal`:

```java
    private static void generation(StringBuilder md, GenerationReport generation) {
        RetrievalOptions retrieval = generation.retrieval();
        md.append("\n## Generation\n\n")
                .append("Answer model `").append(generation.answerModel()).append("`, judge `")
                .append(generation.judgeModel()).append("`. Every question went through the chat path with its defaults: ")
                .append(retrieval.mode().name().toLowerCase(Locale.ROOT)).append(" retrieval, top ")
                .append(retrieval.topK())
                .append(retrieval.rerank() ? ", rerank on, min-score " + number(retrieval.minScore()) : ", rerank off")
                .append(".\n\n")
                .append("| Questions | Count | Answered | Refused by retrieval | Refused by model | Failed | Faithful ")
                .append("| Relevant | Correct | Citations valid |\n|---|---|---|---|---|---|---|---|---|---|\n");
        groupRow(md, "answerable", generation.answerable());
        groupRow(md, "unanswerable", generation.unanswerable());
        md.append("\nFaithful, Relevant and Correct are judge verdicts on the answered questions; Correct needs a reference ")
                .append("answer, so it covers answerable questions only. p50 generation: ")
                .append(generation.p50GenerationMillis()).append(" ms. Judge errors (not counted as fails): ")
                .append(generation.judgeErrors()).append(".\n");

        md.append("\n### Generation: per question\n\n")
                .append("| Id | Outcome | Sources | Faithful | Relevant | Correct | Citations | Question |\n")
                .append("|---|---|---|---|---|---|---|---|\n");
        for (Item item : generation.items()) {
            md.append("| ").append(item.id())
                    .append(" | ").append(outcome(item.outcome()))
                    .append(" | ").append(item.sources())
                    .append(" | ").append(verdict(item.faithful()))
                    .append(" | ").append(verdict(item.relevant()))
                    .append(" | ").append(verdict(item.correct()))
                    .append(" | ").append(item.citations() == null ? "–" : item.citations().valid() ? "✓" : "✗")
                    .append(" | ").append(cell(item.question())).append(" |\n");
        }

        md.append("\n### Answers to review\n\n");
        List<Item> toReview = generation.items().stream().filter(item -> !reviewReasons(item).isEmpty()).toList();
        if (toReview.isEmpty()) {
            md.append("None.\n");
        }
        for (Item item : toReview) {
            String answer = oneLine(item.answer());
            md.append("- **").append(item.id()).append("** ").append(oneLine(item.question())).append(" — ")
                    .append(String.join("; ", reviewReasons(item))).append('\n')
                    .append("  > ").append(answer.length() > 300 ? answer.substring(0, 300) + "…" : answer).append('\n');
        }
    }

    private static void groupRow(StringBuilder md, String name, Group group) {
        md.append("| ").append(name)
                .append(" | ").append(group.questions())
                .append(" | ").append(group.answered())
                .append(" | ").append(group.refusedByRetrieval())
                .append(" | ").append(group.refusedByModel())
                .append(" | ").append(group.failed())
                .append(" | ").append(rate(group.faithful()))
                .append(" | ").append(rate(group.relevant()))
                .append(" | ").append(rate(group.correct()))
                .append(" | ").append(rate(group.citationsValid())).append(" |\n");
    }

    /** Why an answer deserves a look: a wrong refusal, an answered unanswerable question, a failed check. */
    private static List<String> reviewReasons(Item item) {
        List<String> reasons = new ArrayList<>();
        if (item.outcome() == Outcome.FAILED) {
            reasons.add("generation failed");
        }
        else if (item.outcome() != Outcome.ANSWERED) {
            if (item.answerable()) {
                reasons.add("refused an answerable question (" + outcome(item.outcome()) + ")");
            }
        }
        else {
            if (!item.answerable()) {
                reasons.add("answered an unanswerable question");
            }
            check(reasons, "faithfulness", item.faithful());
            check(reasons, "relevancy", item.relevant());
            check(reasons, "correctness", item.correct());
            if (!item.citations().valid()) {
                reasons.add(item.citations().cited().isEmpty() ? "no citations"
                        : "citations out of range: " + item.citations().outOfRange().stream().map(String::valueOf)
                                .collect(joining(", ")));
            }
        }
        return reasons;
    }

    private static void check(List<String> reasons, String name, Verdict verdict) {
        if (verdict == Verdict.FAIL) {
            reasons.add(name + " failed");
        }
        else if (verdict == Verdict.ERROR) {
            reasons.add(name + " judge error");
        }
    }

    private static String outcome(Outcome outcome) {
        return switch (outcome) {
            case ANSWERED -> "answered";
            case REFUSED_BY_RETRIEVAL -> "refused by retrieval";
            case REFUSED_BY_MODEL -> "refused by model";
            case FAILED -> "failed";
        };
    }

    private static String verdict(Verdict verdict) {
        if (verdict == null) {
            return "–";
        }
        return switch (verdict) {
            case PASS -> "✓";
            case FAIL -> "✗";
            case ERROR -> "error";
        };
    }

    private static String rate(Rate rate) {
        return rate.judged() == 0 ? "–" : rate.passed() + "/" + rate.judged();
    }

    private static String number(double value) {
        return value == Math.rint(value) ? String.valueOf((long) value) : String.valueOf(value);
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `./mvnw -q test -Dtest=ReportWriterTest`
Expected: PASS.

- [ ] **Step 6: Document how to run it**

In `eval/README.md`, at the end of section `## 3. Run the eval` (before `## Metrics`), add:

````markdown
To also measure the answers, add the generation flag. It calls the answer model and the judge for every question, so it
takes about 20 more minutes and costs well under a dollar:

```bash
./mvnw spring-boot:run -Dspring-boot.run.profiles=eval -Dspring-boot.run.arguments=--rag.eval.generation=true
```

Every question goes through the chat path with the chat defaults. Each answer is classified, then judged:
- **Classification:** answered, refused by retrieval (no sources), refused by the model (the prompt's "I couldn't
  find…" sentence), or failed.
- **Judging:** only answered questions are judged, by Spring AI's evaluators on the utility model (`gpt-4.1-mini`), a
  different model from the one that answers.
````

In `## Metrics`, add these rows at the end of the table:

```markdown
| Faithful | Answered questions whose answer the judge finds supported by its sources (`FactCheckingEvaluator`: the answer is the claim, the sources are the document). |
| Relevant | Answered questions whose answer responds to the question in line with the sources (`RelevancyEvaluator`). |
| Correct | Answered questions whose answer contains the golden `referenceAnswer` (`FactCheckingEvaluator`: the reference is the claim, the answer is the document). Answerable questions only. |
| Citations valid | Answered questions that cite at least one source, with every `[n]` within 1..sources (`CitationValidator`; numbers in code don't count). |
```

- [ ] **Step 7: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval eval/README.md
git commit -m "feat: eval --rag.eval.generation=true adds judged answers, refusals and citation validity to the report"
```

---

### Task 7: Measure the answers, check the endpoints end to end, document (M7)

**Files:**
- Modify: `eval/README.md`, `README.md`, `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`
- Create: `docs/images/07-debug-trace.png`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Run the eval with generation (calls OpenAI)**

The run takes about 35 minutes: the six retrieval configs, then 73 answers and about 200 judge calls. Run it in the background and keep the Mac awake.

Run: `caffeinate -i ./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval -Dspring-boot.run.arguments=--rag.eval.generation=true`
Expected:
- six retrieval summary lines;
- 73 `Generation <id> (n/73): …` lines;
- `Report written to eval/reports/<time>.md`.

The report has a `## Generation` section with an `answerable` row of 63 and an `unanswerable` row of 10. Judge errors should be 0 or close to it.

- [ ] **Step 2: Check `/api/retrieve` end to end (spec check 4; calls OpenAI for query embeddings)**

Start the app with `./mvnw -q spring-boot:run`. Then run each command:

Run: `curl -s -X POST localhost:8081/api/retrieve -H 'Content-Type: application/json' -d '{"question":"spring.ai.vectorstore.pgvector.index-type","mode":"VECTOR","rerank":false,"topK":10}' | python3 -c "import json,sys; r=json.load(sys.stdin); print([c['sourcePath']+' › '+c['breadcrumb'] for c in r['chunks']][:5]); print([s['name'] for s in r['trace']['stages']])"`
Expected:
- chunks, and the stages `['vector']`.

Then rerun the same command with `"mode":"HYBRID"`.
Expected:
- the stages `['vector', 'keyword', 'fusion']`;
- the PGvector `Configuration properties` chunk at rank 1, or higher than in VECTOR mode (the spec's identifier check).

Finally, rerun it with `"topK":21`.
Expected: HTTP 400, as a problem detail.

- [ ] **Step 3: Check the debug panel in the browser**

At `http://localhost:8081`, tick **Debug trace** and ask "How do I configure the HNSW index for PGvector?". Under the sources, the **Retrieval trace** must list the stages `vector`, `keyword`, `fusion` and `rerank`:
- each stage summary shows its time and chunk count;
- opening `rerank` shows 20 chunks with `N/10` ratings;
- opening `fusion` shows 4-decimal scores.

Expand `fusion` and `rerank`, then capture `docs/images/07-debug-trace.png` at 1280 px wide, full page. Untick Debug, ask again, and check that no trace appears. Stop the app.

- [ ] **Step 4: Record the results**

In `eval/README.md`, add a section `## M7: answer quality` directly above `## M6: reranking and refusals`. It contains:
- the report path and the Generation summary table, copied verbatim, with its notes line;
- **2–4 bullets on:**
  - the false refusals (answerable questions refused, and by what);
  - which unanswerable questions got an answer, and how the model handled the two that M6 let through (watsonx, Couchbase);
  - the faithfulness, relevancy and correctness failures, with one or two examples quoted from "Answers to review";
  - citation validity;
- a caveat: the judge is one LLM call per check with a yes/no verdict; the golden set is AI-written and AI-reviewed; it is one run.

In `README.md`:
1. **Status:** set M7 to `✅ done (…)` with a short outcome, and M8 to `next (optional)`.
2. **Using the app:** add a short `### Debug trace` subsection after `### When the docs don't contain the answer`. It describes the toggle and shows `![The retrieval trace: every stage with its time and ranked chunks, the reranker's ratings expanded](docs/images/07-debug-trace.png)`.
3. **Evaluation:**
   - after the M6 paragraph, add the generation command and an `**M7 answer quality (…):**` table (the Generation summary), followed by one sentence;
   - change the link sentence to point at `eval/README.md#m7-answer-quality` as well.
4. **Known limitations:** add one bullet. The answer judge is one yes/no LLM call per check; Spring AI counts only an exact "yes" as a pass; refusals are recognised by their sentence.
5. **Roadmap:** remove the M7 bullet.

In the spec, append an **Outcome** sub-list to amendment 39: the Generation summary numbers (answered, refused by retrieval and model, faithful, relevant, correct, citations valid, judge errors) and the report path.

- [ ] **Step 5: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add eval/README.md README.md docs/superpowers/specs/2026-10-05-rag-pipeline-design.md docs/images/07-debug-trace.png
git commit -m "eval: M7 answer quality baseline; debug trace documented"
```

---

## After this plan

M7 is complete when Task 7 records the Generation baseline, `/api/retrieve` passes the spec's check 4, and the debug panel shows the trace.

Next is **M8, which is optional: an `ollama` profile.** It adds local `nomic-embed-text` embeddings (768-d, with a `V2` table `vector_store_768`) and a small local chat model, then runs the same retrieval and generation evals on the local profile. It gets its own plan if wanted.
