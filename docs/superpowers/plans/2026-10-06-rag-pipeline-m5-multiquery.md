# RAG Pipeline M5: Query Rewriting and Multi-Query Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional query rewriting and multi-query expansion in front of hybrid retrieval: each query variant is searched in parallel and the results are fused across variants. The eval adds rows that show what each costs in latency and whether it helps. Defaults change only if the pre-registered rule says so.

**Architecture:**
- **Utility client:** a separate `utilityChatClient` (in `AiConfig`) runs `gpt-4.1-mini` at temperature 0 with no advisors. It serves:
  - `QueryRewriter`, a wrapper around Spring AI's `RewriteQueryTransformer`;
  - `LlmQueryExpander`, our own `QueryExpander`, which uses structured output.
- **`RetrievalPipeline`:**
  1. optionally rewrites the question;
  2. optionally expands it into the original plus N variants;
  3. searches every query in parallel on virtual threads, each in the configured mode;
  4. joins the per-query rankings with RRF (k = 60) into the top K.
- **Trace:** stages `rewrite`, `expand`, `q1:vector` … `qN:fusion`, and `join`, with the queries that were actually searched.
- **Failures:** if rewriting or expansion fails, retrieval falls back to the original question, and the eval records how many queries were really searched.

**Tech Stack:** Java 25, Spring Boot 4.1.1, Spring AI 2.0.1 (`RewriteQueryTransformer`, the `QueryExpander` interface, `ChatClient` structured output, `ChatOptions`), OpenAI `gpt-4.1-mini` for utility calls, virtual threads, JUnit 5, Mockito, Testcontainers.

**Spec:** `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`. This plan covers milestone M5 and the Architecture steps ① rewrite, ② expand, ③ retrieve per query and ④ join. It adds amendments 28–32.

## Global Constraints

- Java 25; Spring Boot `4.1.1`; `spring-ai-bom` `2.0.1`; `./mvnw`. Package `com.learnings.rag`; `rag.*` configuration records.
- Jackson 3 only. Prompts live in `src/main/resources/prompts/*.st` and are passed as Message objects, never as template strings.
- The utility client is separate from the answer client and has no advisors (spec "Cross-cutting").
- `*Test` = unit tests without Docker; `*IT` = Testcontainers ITs. Tests never call OpenAI; `StubChatModel` stands in.
- Spec values: RRF k = 60; multi-query = original + 3 variants; candidates per retriever = 20.
- Learning project: no auth or rate limits. App on `127.0.0.1:8081`. Work on branch `m5-multiquery`.

**Spec amendments made while planning (also recorded in the spec):**
28. **Utility ChatClient** (`AiConfig.utilityChatClient`) uses model `rag.retrieval.utility-model` (default `gpt-4.1-mini`) at temperature 0, with no advisors.
    - Probe: `gpt-5-mini` rejects `temperature: 0` ("Only the default (1) value is supported") and is a slower reasoning model. `gpt-4.1-mini` rewrote a chatty question in 1.4 s.
    - The answer client keeps `OPENAI_CHAT_MODEL`.
29. **Rewriting uses Spring AI's `RewriteQueryTransformer`. Expansion uses our own `LlmQueryExpander implements QueryExpander`**, with structured output.
    - Why not Spring AI's `MultiQueryExpander`: it splits the reply on newlines and silently returns only the original when the count differs (for example, a blank line between variants). Multi-query would then be invisibly off.
    - The original question always comes first. Blank, duplicate and original-copy variants are dropped, and at most N are kept.
    - Model failures fall back to the original question, and the eval reports the queries actually searched.
30. **Multi-query flow:**
    1. rewrite (optional);
    2. expand (optional): the question plus N variants;
    3. search every query in parallel in the configured mode, each to depth `candidates`;
    4. RRF across the queries (k = 60), keeping the top K.

    A single query keeps M4's stage names exactly.
31. **Golden set v3** adds 10 conversational items (`c01`–`c10`, tag `conversational`).
    - They are chatty, vague rephrasings of verified items, and they copy those items' labels.
    - They were written by Claude.
    - M5 compares configurations on these 63 items.
32. **Pre-registered default rule.**
    - Candidates: `hybrid+rewrite`, `hybrid+multiquery`, `hybrid+rewrite+multiquery`, each compared with `hybrid` in the same run.
    - A candidate qualifies only if both hold:
      - its MRR@10 is at least hybrid's + 1/N;
      - its hit@5 is at least hybrid's − 1/N, where N = 63.
    - If several qualify, the highest MRR@10 wins; on a tie, the one with fewer LLM calls wins.
    - The winner's `rewrite` and `query-variants` become the defaults. If none qualifies, both stay off.
    - Latency is reported, not capped.

## Review Focus

Inputs the spec implies but doesn't spell out, most likely to bite first. Each one gets a test in the task that owns it:

1. **Messy expansion output** must still yield the original plus at most N distinct variants, and never silently zero variants. Messy means numbering, bullets, blank entries, case-only duplicates, a copy of the original, more than N items, or unparseable JSON. Covered by `cleansMessyVariants`, `unparseableReplyMeansTheOriginalOnly` (Task 2), and the eval's queries-searched line (Task 4).
2. **A utility-model failure** during rewriting or expansion (outage, 429, a bad model name) must not fail retrieval. The original question is searched instead, and the failure is logged. Covered by `aFailingModelKeepsTheOriginalQuestion` and `aFailingModelMeansTheOriginalOnly` (Task 2).
3. **Braces in the question** (`{name}`) must reach the model intact through `RewriteQueryTransformer`'s prompt template. Covered by `bracesInTheQuestionReachTheModelIntact` (Task 2).
4. **A failing per-variant search** must fail the request with its own exception, like M4's parallel retrievers. Covered by `aFailingVariantSearchFailsWithItsOwnException` (Task 3).
5. **A chunk found by several variants** must appear once in the joined list, ranked by its summed RRF score. Covered by `variantsAreSearchedInParallelAndJoinedWithoutDuplicates` (Task 3).

---

## File map

```
src/main/java/com/learnings/rag/
  config/AiConfig.java                         utilityChatClient bean (new)
  config/RagProperties.java                    Retrieval + rewrite, queryVariants, utilityModel (modify)
  retrieval/QueryRewriter.java                 RewriteQueryTransformer wrapper with fallback (new)
  retrieval/LlmQueryExpander.java              QueryExpander via structured output (new)
  retrieval/RetrievalOptions.java              + rewrite, queryVariants (modify)
  retrieval/PipelineTrace.java                 Stage.queries, queries() (modify)
  retrieval/RetrievalPipeline.java             rewrite → expand → parallel per-query search → join (modify)
  eval/EvalReport.java, EvalRunner.java, ReportWriter.java, EvalConfig.java   queries searched; M5 configs (modify)
src/main/resources/
  prompts/query-expansion.st (new), application.yml (rag.retrieval.rewrite / query-variants / utility-model)
  static/app.js                                condensed scores for multi-query sources
src/test/java/com/learnings/rag/
  retrieval/{QueryRewriterTest, LlmQueryExpanderTest}.java (new); retrieval/{RetrievalPipelineTest, RetrievalOptionsTest}.java (modify)
  eval/{ReportWriterTest, EvalRunnerIT}.java (modify)
eval/golden-set.json, eval/README.md, README.md, spec
```

---

### Task 1: Conversational questions in the golden set (M5)

**Files:**
- Modify: `eval/golden-set.json` (10 new items), `eval/README.md`

**Interfaces:**
- Consumes: golden items `q03`, `q09`, `q13`, `q19`, `q24`, `q27`, `q29`, `q33`, `q39` and `h04`, whose labels were all verified in M3 and M4.

- [ ] **Step 1: Add the conversational items**

Run this from the repo root:

```bash
python3 - <<'EOF'
import json
path = 'eval/golden-set.json'
items = json.load(open(path))
by_id = {i['id']: i for i in items}
assert not any(i['id'].startswith('c') for i in items), 'conversational items already added'

# Chatty, vague rephrasings of verified items; each copies its source item's labels.
rephrasings = [
    ("c01", "q33", "hey so my pgvector searches feel slow, which index thing am I supposed to pick?"),
    ("c02", "q13", "ollama keeps downloading the model every time I start the app, can I make it only fetch it when it's missing?"),
    ("c03", "q19", "my MCP client blows up at startup because the user isn't logged in yet... any way around that?"),
    ("c04", "q27", "i need my tool to know which customer is calling but the LLM must not see the id, how do I pass it?"),
    ("c05", "q24", "openai throws an error when I ask for a list of records with the native structured output switch, what am I doing wrong"),
    ("c06", "q29", "token counts look way too high after tool calls - does getUsage add up every call or just the last one?"),
    ("c07", "q09", "where does the tool calling advisor sit in the advisor chain by default? my own advisor seems to run after it"),
    ("c08", "q03", "I want to change how the long-term memory gets stuffed into the system prompt with the vector store memory advisor"),
    ("c09", "h04", "don't want to install postgres on my laptop, how do I get pgvector running for local dev?"),
    ("c10", "q39", "do embeddings get metrics and tracing with every provider or only some of them?"),
]
for new_id, source_id, question in rephrasings:
    source = by_id[source_id]
    items.append({"id": new_id, "question": question,
                  "expectedSources": source["expectedSources"],
                  "referenceAnswer": source.get("referenceAnswer"),
                  "sourceExcerpt": None, "tags": ["conversational"]})
json.dump(items, open(path, 'w'), indent=2, ensure_ascii=False)
open(path, 'a').write('\n')
print(len(items), 'items;', sum('conversational' in (i.get('tags') or []) for i in items), 'tagged conversational')
EOF
```

Expected: `63 items; 10 tagged conversational`.

- [ ] **Step 2: Validate against the index (calls OpenAI for query embeddings)**

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval`
Expected:
- the run completes;
- the report's `## By tag` table has `conversational | 10`, `identifier | 13` and `untagged | 40` rows for vector, keyword and hybrid.

Note how hybrid scores on `conversational`: that is the before picture for M5.

- [ ] **Step 3: Document v3 in `eval/README.md`**

Add this bullet under `## Golden set versions`, after the v2 bullet:

```markdown
- **v3 (M5, 63 items):** adds `c01`–`c10`. Each is a chatty, vague rephrasing of a verified item (typos, filler,
  symptoms instead of terms), with that item's labels and `"tags": ["conversational"]`. They are written by Claude to
  test query rewriting and multi-query expansion, which target exactly this kind of question.
```

- [ ] **Step 4: Commit**

```bash
git add eval/golden-set.json eval/README.md
git commit -m "eval: 10 conversational rephrasings tagged for per-tag reporting (golden set v3)"
```

---

### Task 2: Utility client, query rewriter and query expander (M5)

**Files:**
- Create: `src/main/java/com/learnings/rag/config/AiConfig.java`, `src/main/java/com/learnings/rag/retrieval/{QueryRewriter, LlmQueryExpander}.java`, `src/main/resources/prompts/query-expansion.st`
- Modify: `src/main/java/com/learnings/rag/config/RagProperties.java`, `src/main/resources/application.yml`
- Test: `src/test/java/com/learnings/rag/retrieval/{QueryRewriterTest, LlmQueryExpanderTest}.java` (new); `RetrievalOptionsTest.java` and `RetrievalPipelineTest.java` (constructor call sites)

**Interfaces:**
- Produces:
  - `RagProperties.Retrieval(int topK, double similarityThreshold, RetrievalMode mode, int candidates, boolean rewrite, int queryVariants, String utilityModel)`, with defaults `false`, `0` and `gpt-4.1-mini` for the new fields.
  - `@Bean ChatClient utilityChatClient` in `AiConfig`.
  - `@Component QueryRewriter(ChatClient utilityChatClient)` with `String rewrite(String question)`, which never throws.
  - `@Component LlmQueryExpander(ChatClient utilityChatClient, Resource systemPrompt, RagProperties) implements QueryExpander`:
    - `List<Query> expand(Query)` uses the configured count;
    - `List<Query> expand(Query, int variants)` puts the original first, never throws, and has a public nested `record QueryVariants(List<String> queries)`.

- [ ] **Step 1: Write the failing tests**

`src/test/java/com/learnings/rag/retrieval/QueryRewriterTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

import com.learnings.rag.StubChatModel;

class QueryRewriterTest {

    private static QueryRewriter rewriter(ChatModel model) {
        return new QueryRewriter(ChatClient.builder(model).build());
    }

    @Test
    void searchesWithTheModelsRewrite() {
        assertThat(rewriter(new StubChatModel("  pgvector index type options \n"))
                .rewrite("hey so which index thing do I pick for pgvector??"))
                .isEqualTo("pgvector index type options");
    }

    @Test
    void anEmptyReplyKeepsTheOriginalQuestion() {
        assertThat(rewriter(new StubChatModel("")).rewrite("Which index types does PGvector support?"))
                .isEqualTo("Which index types does PGvector support?");
    }

    @Test
    void aFailingModelKeepsTheOriginalQuestion() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("429 Too Many Requests");
            }
        };

        assertThat(rewriter(unavailable).rewrite("Which index types does PGvector support?"))
                .isEqualTo("Which index types does PGvector support?");
    }

    @Test
    void bracesInTheQuestionReachTheModelIntact() {
        StubChatModel model = new StubChatModel("template placeholders");

        rewriter(model).rewrite("How do I fill {name} in a PromptTemplate?");

        assertThat(model.prompts()).singleElement().satisfies(prompt ->
                assertThat(prompt.getUserMessage().getText()).contains("How do I fill {name} in a PromptTemplate?"));
    }
}
```

`src/test/java/com/learnings/rag/retrieval/LlmQueryExpanderTest.java`:

```java
package com.learnings.rag.retrieval;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.rag.Query;
import org.springframework.core.io.ClassPathResource;

import com.learnings.rag.StubChatModel;
import com.learnings.rag.config.RagProperties;

class LlmQueryExpanderTest {

    private static final Query QUESTION = new Query("Which index types does PGvector support?");

    private static LlmQueryExpander expander(ChatModel model) {
        RagProperties properties = new RagProperties(Path.of("corpus"), "model", new RagProperties.Chunking(500, 50, 60),
                new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 3, "gpt-4.1-mini"));
        return new LlmQueryExpander(ChatClient.builder(model).build(),
                new ClassPathResource("prompts/query-expansion.st"), properties);
    }

    private static List<String> texts(List<Query> queries) {
        return queries.stream().map(Query::text).toList();
    }

    @Test
    void theOriginalComesFirstThenTheVariants() {
        StubChatModel model = new StubChatModel(
                "{\"queries\": [\"pgvector HNSW vs IVFFlat\", \"spring.ai.vectorstore.pgvector.index-type\", \"postgres vector index options\"]}");

        assertThat(texts(expander(model).expand(QUESTION))).containsExactly(QUESTION.text(),
                "pgvector HNSW vs IVFFlat", "spring.ai.vectorstore.pgvector.index-type", "postgres vector index options");
    }

    @Test
    void cleansMessyVariants() {
        StubChatModel model = new StubChatModel("{\"queries\": [\"1. pgvector HNSW index\", \"\", \"PGvector  hnsw index\", "
                + "\"Which index types does PGvector support?\", \"- index type property\", \"  another phrasing \", \"one too many\"]}");

        assertThat(texts(expander(model).expand(QUESTION, 3))).containsExactly(QUESTION.text(),
                "pgvector HNSW index", "index type property", "another phrasing");
    }

    @Test
    void unparseableReplyMeansTheOriginalOnly() {
        assertThat(texts(expander(new StubChatModel("here are some ideas: a, b, c")).expand(QUESTION)))
                .containsExactly(QUESTION.text());
    }

    @Test
    void aFailingModelMeansTheOriginalOnly() {
        ChatModel unavailable = new ChatModel() {
            @Override
            public ChatResponse call(Prompt prompt) {
                throw new IllegalStateException("model not found");
            }
        };

        assertThat(texts(expander(unavailable).expand(QUESTION))).containsExactly(QUESTION.text());
    }

    @Test
    void zeroVariantsSkipsTheModel() {
        StubChatModel model = new StubChatModel("{\"queries\": [\"x\"]}");

        assertThat(texts(expander(model).expand(QUESTION, 0))).containsExactly(QUESTION.text());
        assertThat(model.prompts()).isEmpty();
    }

    @Test
    void thePromptAsksForTheRequestedNumberAndCarriesTheQuestion() {
        StubChatModel model = new StubChatModel("{\"queries\": []}");

        expander(model).expand(QUESTION, 3);

        assertThat(model.prompts()).singleElement().satisfies(prompt -> {
            assertThat(prompt.getSystemMessage().getText()).contains("Spring AI reference documentation");
            assertThat(prompt.getUserMessage().getText()).contains("Write 3 alternative search queries", QUESTION.text());
        });
    }
}
```

Update the constructor call sites of `RagProperties.Retrieval` in the existing tests:
- In `src/test/java/com/learnings/rag/retrieval/RetrievalOptionsTest.java`, change `new RagProperties.Retrieval(topK, threshold, mode, candidates)` to `new RagProperties.Retrieval(topK, threshold, mode, candidates, false, 0, "gpt-4.1-mini")`.
- In `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineTest.java`, change `new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20)` to `new RagProperties.Retrieval(5, 0.0, RetrievalMode.HYBRID, 20, false, 0, "gpt-4.1-mini")`.

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest='QueryRewriterTest,LlmQueryExpanderTest'`
Expected: compilation FAILURE (`cannot find symbol: class QueryRewriter`; `Retrieval` has no 7-argument constructor).

- [ ] **Step 3: Implement the utility client, the rewriter and the expander**

In `src/main/java/com/learnings/rag/config/RagProperties.java`, replace the `Retrieval` record and its javadoc with:

```java
    /**
     * @param topK number of chunks handed to the model
     * @param similarityThreshold minimum cosine similarity for vector search; 0 keeps every positive similarity
     * @param mode which retrievers run: VECTOR, KEYWORD, or HYBRID (both, fused with reciprocal rank fusion)
     * @param candidates how many chunks each retriever contributes before fusion (in HYBRID mode and per query variant)
     * @param rewrite rewrite the question with the utility model before searching
     * @param queryVariants extra phrasings of the question searched as well and fused across queries; 0 turns it off
     * @param utilityModel chat model for rewriting and expansion; it must accept temperature 0 (gpt-5 models do not)
     */
    public record Retrieval(@DefaultValue("5") int topK,
            @DefaultValue("0.0") double similarityThreshold,
            @DefaultValue("VECTOR") RetrievalMode mode,
            @DefaultValue("20") int candidates,
            @DefaultValue("false") boolean rewrite,
            @DefaultValue("0") int queryVariants,
            @DefaultValue("gpt-4.1-mini") String utilityModel) {
    }
```

In `src/main/resources/application.yml`, add these lines directly below `    candidates: 20 …` in the `rag.retrieval` block:

```yaml
    rewrite: false                # rewrite the question with the utility model first (decided by the M5 eval)
    query-variants: 0             # extra phrasings searched and fused across queries; 0 = off (M5 eval)
    utility-model: gpt-4.1-mini   # rewriting/expansion; must accept temperature 0
```

`src/main/java/com/learnings/rag/config/AiConfig.java`:

```java
package com.learnings.rag.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration(proxyBeanMethods = false)
public class AiConfig {

    /**
     * The utility client for query rewriting and expansion (and later reranking and judging): a small, fast,
     * deterministic model, separate from the answer client, with no advisors. The answer client is built in
     * AnswerService from its own ChatClient.Builder, which is prototype-scoped, so these options never reach it.
     */
    @Bean
    ChatClient utilityChatClient(ChatClient.Builder builder, RagProperties properties) {
        return builder
                .defaultOptions(ChatOptions.builder()
                        .model(properties.retrieval().utilityModel())
                        .temperature(0.0))
                .build();
    }
}
```

`src/main/resources/prompts/query-expansion.st`:

```
You write alternative search queries for a search engine over the Spring AI reference documentation (Java, Spring Boot).

Given a developer's question, write the requested number of alternative queries that could find the documentation section that answers it.

Rules:
- Keep the intent of the original question. Do not answer it.
- Vary the wording: use likely documentation terms, the Spring AI class, annotation or property names the answer probably involves, and one plain-language phrasing.
- Each query is a single line of at most 20 words, with no numbering.
- Return exactly the requested number of queries.
```

`src/main/java/com/learnings/rag/retrieval/QueryRewriter.java`:

```java
package com.learnings.rag.retrieval;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.transformation.RewriteQueryTransformer;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Component;

/** Rewrites a chatty or vague question into a concise search query with Spring AI's RewriteQueryTransformer. */
@Component
public class QueryRewriter {

    private static final Logger log = LoggerFactory.getLogger(QueryRewriter.class);

    private final RewriteQueryTransformer transformer;

    public QueryRewriter(@Qualifier("utilityChatClient") ChatClient utilityChatClient) {
        this.transformer = RewriteQueryTransformer.builder()
                .chatClientBuilder(utilityChatClient.mutate())
                .targetSearchSystem("search engine over the Spring AI reference documentation")
                .build();
    }

    /** The rewritten question; the original when the model fails or answers with nothing (logged, never thrown). */
    public String rewrite(String question) {
        try {
            String rewritten = transformer.transform(new Query(question)).text();
            return rewritten == null || rewritten.isBlank() ? question : rewritten.strip();
        }
        catch (RuntimeException e) {
            log.warn("Query rewrite failed; searching with the original question", e);
            return question;
        }
    }
}
```

`src/main/java/com/learnings/rag/retrieval/LlmQueryExpander.java`:

```java
package com.learnings.rag.retrieval;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.expansion.QueryExpander;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/**
 * Writes alternative phrasings of a question. Unlike Spring AI's MultiQueryExpander, which splits the reply on
 * newlines and silently returns only the original when the count differs, this asks for structured output and keeps
 * every usable variant: numbering and bullets are stripped; blanks, duplicates and copies of the original are dropped;
 * at most {@code variants} are kept. The original question always comes first, and on any failure it is all that is
 * returned (logged, never thrown).
 */
@Component
public class LlmQueryExpander implements QueryExpander {

    private static final Logger log = LoggerFactory.getLogger(LlmQueryExpander.class);

    /** The structured reply the model is asked for. */
    public record QueryVariants(List<String> queries) {
    }

    private final ChatClient chatClient;
    private final String systemPrompt;
    private final int defaultVariants;

    public LlmQueryExpander(@Qualifier("utilityChatClient") ChatClient utilityChatClient,
            @Value("classpath:prompts/query-expansion.st") Resource systemPrompt, RagProperties properties) {
        this.chatClient = utilityChatClient;
        this.defaultVariants = properties.retrieval().queryVariants();
        try {
            this.systemPrompt = systemPrompt.getContentAsString(StandardCharsets.UTF_8);
        }
        catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public List<Query> expand(Query query) {
        return expand(query, defaultVariants);
    }

    public List<Query> expand(Query query, int variants) {
        if (variants < 1) {
            return List.of(query);
        }
        QueryVariants reply;
        try {
            reply = chatClient
                    .prompt(new Prompt(List.of(new SystemMessage(systemPrompt), new UserMessage(
                            "Write " + variants + " alternative search queries for this question:\n" + query.text()))))
                    .call()
                    .entity(QueryVariants.class);
        }
        catch (RuntimeException e) {
            log.warn("Query expansion failed; searching with the original question only", e);
            return List.of(query);
        }

        List<Query> queries = new ArrayList<>();
        queries.add(query);
        Set<String> seen = new HashSet<>();
        seen.add(normalize(query.text()));
        if (reply != null && reply.queries() != null) {
            for (String candidate : reply.queries()) {
                if (queries.size() > variants) {
                    break;
                }
                String text = candidate == null ? "" : candidate.strip().replaceFirst("^(\\d+[.)]|[-*•])\\s*", "").strip();
                if (!text.isEmpty() && seen.add(normalize(text))) {
                    queries.add(new Query(text));
                }
            }
        }
        if (queries.size() == 1) {
            log.warn("Query expansion produced no usable variant for: {}", query.text());
        }
        return List.copyOf(queries);
    }

    private static String normalize(String text) {
        return text.strip().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }
}
```

- [ ] **Step 4: Run them and watch them pass**

Run: `./mvnw -q test -Dtest='QueryRewriterTest,LlmQueryExpanderTest,RetrievalOptionsTest,RetrievalPipelineTest'`
Expected: PASS (4 + 6 + 3 + 3 tests).

- [ ] **Step 5: Run every test**

Every IT context now creates `utilityChatClient` over `StubChatModel`.
Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS.

- [ ] **Step 6: Commit**

```bash
git add src/main/java/com/learnings/rag src/main/resources src/test/java/com/learnings/rag
git commit -m "feat: utility chat client, query rewriter and structured-output query expander"
```

---

### Task 3: Rewrite, expand and search every variant in parallel (M5)

**Files:**
- Modify: `src/main/java/com/learnings/rag/retrieval/{RetrievalOptions, PipelineTrace, RetrievalPipeline}.java`, `src/main/resources/static/app.js`
- Test: `src/test/java/com/learnings/rag/retrieval/{RetrievalPipelineTest, RetrievalOptionsTest}.java`

**Interfaces:**
- Consumes: `QueryRewriter.rewrite(String)` and `LlmQueryExpander.expand(Query, int)` (Task 2); `ReciprocalRankFusion` (M4).
- Produces:
  - `RetrievalOptions`:
    - components `(int topK, double similarityThreshold, RetrievalMode mode, int candidates, boolean rewrite, int queryVariants)`;
    - a 4-argument constructor means no rewrite and no variants;
    - `withRewrite(boolean)` and `withQueryVariants(int)`, with `queryVariants >= 0`.
  - `PipelineTrace`:
    - `Stage(String name, long elapsedMillis, List<Hit> hits, List<String> queries)`, plus a 3-argument constructor with no queries;
    - `List<String> queries()`, which returns the queries of the last stage that lists any, or an empty list.
  - `RetrievalPipeline(VectorRetriever, KeywordRetriever, QueryRewriter, LlmQueryExpander, RagProperties)`:
    - a single query keeps M4's stage names;
    - several queries produce stages `q{i}:…` and a final `join`.

- [ ] **Step 1: Write the failing tests**

In `src/test/java/com/learnings/rag/retrieval/RetrievalOptionsTest.java`, add:

```java
    @Test
    void rewriteAndVariantsDefaultToOffAndCanBeSetPerCall() {
        RetrievalOptions options = new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20);

        assertThat(options.rewrite()).isFalse();
        assertThat(options.queryVariants()).isZero();
        assertThat(options.withRewrite(true).withQueryVariants(3))
                .isEqualTo(new RetrievalOptions(5, 0.0, RetrievalMode.HYBRID, 20, true, 3));
        assertThatThrownBy(() -> options.withQueryVariants(-1)).hasMessageContaining("queryVariants");
    }
```

In `src/test/java/com/learnings/rag/retrieval/RetrievalPipelineTest.java`, make these changes:

1. Add the mocks below the existing two:

```java
    private final QueryRewriter queryRewriter = mock(QueryRewriter.class);
    private final LlmQueryExpander queryExpander = mock(LlmQueryExpander.class);
```

2. Change the pipeline construction to `new RetrievalPipeline(vectorRetriever, keywordRetriever, queryRewriter, queryExpander, new RagProperties(...))`, keeping the existing `RagProperties` argument.
3. Add the imports `org.springframework.ai.rag.Query`, `java.util.Map` and `static org.mockito.ArgumentMatchers.eq`.
4. Add these tests:

```java
    private static RetrievalOptions vectorOnly() {
        return new RetrievalOptions(5, 0.0, RetrievalMode.VECTOR, 20);
    }

    @Test
    void rewriteSearchesTheRewrittenQuestion() {
        when(queryRewriter.rewrite("hey which index thing for pgvector?")).thenReturn("pgvector index types");
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call ->
                List.of(doc("for: " + ((Query) call.getArgument(0)).text())));

        RetrievalResult result = pipeline.retrieve("hey which index thing for pgvector?", vectorOnly().withRewrite(true));

        assertThat(result.documents()).extracting(Document::getId).containsExactly("for: pgvector index types");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("rewrite", "vector");
        assertThat(result.trace().queries()).containsExactly("pgvector index types");
    }

    @Test
    void variantsAreSearchedInParallelAndJoinedWithoutDuplicates() {
        when(queryExpander.expand(any(), eq(2))).thenReturn(
                List.of(new Query("original"), new Query("variant one"), new Query("variant two")));
        Map<String, List<Document>> rankings = Map.of(
                "original", List.of(doc("shared"), doc("a")),
                "variant one", List.of(doc("b"), doc("shared")),
                "variant two", List.of(doc("shared"), doc("c")));
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call ->
                rankings.get(((Query) call.getArgument(0)).text()));

        RetrievalResult result = pipeline.retrieve("original", vectorOnly().withQueryVariants(2));

        assertThat(result.documents()).extracting(Document::getId).startsWith("shared").doesNotHaveDuplicates()
                .containsExactlyInAnyOrder("shared", "a", "b", "c");
        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name)
                .containsExactly("expand", "q1:vector", "q2:vector", "q3:vector", "join");
        assertThat(result.trace().queries()).containsExactly("original", "variant one", "variant two");
    }

    @Test
    void aFailingVariantSearchFailsWithItsOwnException() {
        when(queryExpander.expand(any(), eq(2))).thenReturn(
                List.of(new Query("original"), new Query("variant one"), new Query("variant two")));
        when(vectorRetriever.retrieve(any(), any())).thenAnswer(call -> {
            if (((Query) call.getArgument(0)).text().equals("variant two")) {
                throw new IllegalStateException("embedding service unavailable");
            }
            return List.of(doc("a"));
        });

        assertThatThrownBy(() -> pipeline.retrieve("original", vectorOnly().withQueryVariants(2)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessage("embedding service unavailable");
    }

    @Test
    void anExpansionThatFellBackToTheOriginalSearchesOnceWithM4StageNames() {
        when(queryExpander.expand(any(), eq(3))).thenReturn(List.of(new Query("original")));
        when(vectorRetriever.retrieve(any(), any())).thenReturn(List.of(doc("a")));

        RetrievalResult result = pipeline.retrieve("original", vectorOnly().withQueryVariants(3));

        assertThat(result.trace().stages()).extracting(PipelineTrace.Stage::name).containsExactly("expand", "vector");
        assertThat(result.trace().queries()).containsExactly("original");
    }
```

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest='RetrievalOptionsTest,RetrievalPipelineTest'`
Expected: compilation FAILURE (`cannot find symbol: method withRewrite`; the `RetrievalPipeline` constructor doesn't match).

- [ ] **Step 3: Implement the multi-query flow**

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
 * @param candidates how many chunks each retriever contributes before fusion (in HYBRID mode and per query variant)
 * @param rewrite rewrite the question with the utility model before searching
 * @param queryVariants extra phrasings searched as well and joined across queries; 0 turns it off
 */
public record RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates,
        boolean rewrite, int queryVariants) {

    public RetrievalOptions {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be at least 1, was " + topK);
        }
        if (candidates < 1) {
            throw new IllegalArgumentException("candidates must be at least 1, was " + candidates);
        }
        if (queryVariants < 0) {
            throw new IllegalArgumentException("queryVariants must not be negative, was " + queryVariants);
        }
        Objects.requireNonNull(mode, "mode");
    }

    /** No rewriting and no query variants. */
    public RetrievalOptions(int topK, double similarityThreshold, RetrievalMode mode, int candidates) {
        this(topK, similarityThreshold, mode, candidates, false, 0);
    }

    public static RetrievalOptions from(RagProperties properties) {
        RagProperties.Retrieval retrieval = properties.retrieval();
        return new RetrievalOptions(retrieval.topK(), retrieval.similarityThreshold(), retrieval.mode(),
                retrieval.candidates(), retrieval.rewrite(), retrieval.queryVariants());
    }

    public RetrievalOptions withTopK(int topK) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withMode(RetrievalMode mode) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withRewrite(boolean rewrite) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }

    public RetrievalOptions withQueryVariants(int queryVariants) {
        return new RetrievalOptions(topK, similarityThreshold, mode, candidates, rewrite, queryVariants);
    }
}
```

In `src/main/java/com/learnings/rag/retrieval/PipelineTrace.java`, replace the `Stage` record with the following, and add the `queries()` method below `PipelineTrace`'s 1-argument constructor:

```java
    /** @param queries for rewrite and expand stages: the queries they produced */
    public record Stage(String name, long elapsedMillis, List<Hit> hits, List<String> queries) {

        public Stage(String name, long elapsedMillis, List<Hit> hits) {
            this(name, elapsedMillis, hits, List.of());
        }
    }

    /** The queries that were searched: those of the last stage that lists any (expand, else rewrite); else empty. */
    public List<String> queries() {
        for (int i = stages.size() - 1; i >= 0; i--) {
            if (!stages.get(i).queries().isEmpty()) {
                return stages.get(i).queries();
            }
        }
        return List.of();
    }
```

Replace `src/main/java/com/learnings/rag/retrieval/RetrievalPipeline.java` with:

```java
package com.learnings.rag.retrieval;

import java.util.ArrayList;
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
 * Retrieves chunks for a question:
 * <ol>
 * <li>optionally rewrites it;</li>
 * <li>optionally expands it into the original plus variants;</li>
 * <li>searches every query in the configured {@link RetrievalMode}, in parallel on virtual threads;</li>
 * <li>joins several rankings with reciprocal rank fusion.</li>
 * </ol>
 * HYBRID runs vector and keyword search in parallel for each query. Every stage is traced; a single query keeps
 * M4's stage names ({@code vector}, {@code keyword}, {@code fusion}), several get a {@code q1:} prefix and a final
 * {@code join}. Reranking arrives in M6.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final KeywordRetriever keywordRetriever;
    private final QueryRewriter queryRewriter;
    private final LlmQueryExpander queryExpander;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, KeywordRetriever keywordRetriever,
            QueryRewriter queryRewriter, LlmQueryExpander queryExpander, RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.keywordRetriever = keywordRetriever;
        this.queryRewriter = queryRewriter;
        this.queryExpander = queryExpander;
        this.defaults = RetrievalOptions.from(properties);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Timed> stages = new ArrayList<>();

        String searchText = question;
        if (options.rewrite()) {
            Timed rewrite = timedQueries("rewrite", () -> List.of(queryRewriter.rewrite(question)));
            stages.add(rewrite);
            searchText = rewrite.queries().getFirst();
        }
        List<String> queries = List.of(searchText);
        if (options.queryVariants() > 0) {
            String text = searchText;
            Timed expand = timedQueries("expand", () -> queryExpander.expand(new Query(text), options.queryVariants())
                    .stream().map(Query::text).toList());
            stages.add(expand);
            queries = expand.queries();
        }

        List<Document> documents;
        if (queries.size() == 1) {
            List<Timed> search = search(queries.getFirst(), options, options.topK(), "");
            stages.addAll(search);
            documents = search.getLast().documents();
        }
        else {
            List<List<Timed>> perQuery = searchInParallel(queries, options);
            perQuery.forEach(stages::addAll);
            Timed join = timed("join", () -> ReciprocalRankFusion.fuse(
                    perQuery.stream().map(search -> search.getLast().documents()).toList(),
                    ReciprocalRankFusion.DEFAULT_K, options.topK()));
            stages.add(join);
            documents = join.documents();
        }
        PipelineTrace trace = new PipelineTrace(stages.stream().map(Timed::stage).toList(), millisSince(start));
        return new RetrievalResult(documents, trace);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    /** Every query in parallel; each keeps {@code candidates} chunks so the join has depth to fuse. */
    private List<List<Timed>> searchInParallel(List<String> queries, RetrievalOptions options) {
        int depth = Math.max(options.candidates(), options.topK());
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<List<Timed>>> searches = new ArrayList<>();
            for (int i = 0; i < queries.size(); i++) {
                String query = queries.get(i);
                String prefix = "q" + (i + 1) + ":";
                searches.add(executor.submit(() -> search(query, options, depth, prefix)));
            }
            List<List<Timed>> results = new ArrayList<>();
            for (Future<List<Timed>> search : searches) {
                results.add(join(search));
            }
            return results;
        }
    }

    /** One query in the configured mode; the last stage holds its final ranking of at most {@code limit} chunks. */
    private List<Timed> search(String query, RetrievalOptions options, int limit, String prefix) {
        return switch (options.mode()) {
            case VECTOR -> List.of(timed(prefix + "vector",
                    () -> vectorRetriever.retrieve(new Query(query), options.withTopK(limit))));
            case KEYWORD -> List.of(timed(prefix + "keyword", () -> keywordRetriever.retrieve(query, limit)));
            case HYBRID -> hybrid(query, options, limit, prefix);
        };
    }

    private List<Timed> hybrid(String query, RetrievalOptions options, int limit, String prefix) {
        int candidates = Math.max(options.candidates(), limit);
        Timed vector;
        Timed keyword;
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Timed> vectorSearch = executor.submit(() -> timed(prefix + "vector",
                    () -> vectorRetriever.retrieve(new Query(query), options.withTopK(candidates))));
            Future<Timed> keywordSearch = executor.submit(() -> timed(prefix + "keyword",
                    () -> keywordRetriever.retrieve(query, candidates)));
            vector = join(vectorSearch);
            keyword = join(keywordSearch);
        }
        Timed fusion = timed(prefix + "fusion", () -> ReciprocalRankFusion.fuse(
                List.of(vector.documents(), keyword.documents()), ReciprocalRankFusion.DEFAULT_K, limit));
        return List.of(vector, keyword, fusion);
    }

    /** Waits for a parallel search; its own exception is rethrown as is, not wrapped. */
    private static <T> T join(Future<T> search) {
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

    private record Timed(String name, List<Document> documents, List<String> queries, long millis) {

        PipelineTrace.Stage stage() {
            return new PipelineTrace.Stage(name, millis, documents.stream().map(PipelineTrace.Hit::of).toList(),
                    queries);
        }
    }

    private static Timed timed(String name, Supplier<List<Document>> search) {
        long start = System.nanoTime();
        List<Document> documents = search.get();
        return new Timed(name, documents, List.of(), millisSince(start));
    }

    private static Timed timedQueries(String name, Supplier<List<String>> transformation) {
        long start = System.nanoTime();
        List<String> queries = transformation.get();
        return new Timed(name, List.of(), queries, millisSince(start));
    }

    private static long millisSince(long start) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
    }
}
```

In `src/main/resources/static/app.js`, replace the `scoreText` function with:

```js
// The score of each stage that returned the chunk; vector search is shown as "similarity". With several query
// variants the per-variant scores are condensed to the join score and how many variants found the chunk.
function scoreText(s) {
  const entries = Object.entries(s.scores || {});
  if (entries.length === 0) return s.score == null ? '' : `score ${s.score.toFixed(3)}`;
  if ('join' in s.scores) {
    const queries = new Set(entries.filter(([stage]) => stage.includes(':')).map(([stage]) => stage.split(':')[0])).size;
    return `join ${s.scores.join.toFixed(4)} · found by ${queries} ${queries === 1 ? 'query' : 'queries'}`;
  }
  return entries.map(([stage, value]) => `${stage === 'vector' ? 'similarity' : stage} ${value.toFixed(3)}`).join(' · ');
}
```

- [ ] **Step 4: Run the tests and watch them pass**

Run: `./mvnw -q test -Dtest='RetrievalOptionsTest,RetrievalPipelineTest,AnswerServiceTest' && node --check src/main/resources/static/app.js`
Expected: PASS (4 + 7 + 6 tests); the JS check is clean.

- [ ] **Step 5: Run every test**

Run: `./mvnw -q verify`
Expected: all unit tests and ITs PASS (single-query behaviour and stage names are unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/main/java/com/learnings/rag/retrieval src/main/resources/static/app.js src/test/java/com/learnings/rag/retrieval
git commit -m "feat: rewrite and multi-query retrieval with parallel per-variant search and RRF join"
```

---

### Task 4: The eval reports the queries searched and compares the M5 configurations (M5)

**Files:**
- Modify: `src/main/java/com/learnings/rag/eval/{EvalReport, EvalRunner, ReportWriter, EvalConfig}.java`
- Test: `src/test/java/com/learnings/rag/eval/{ReportWriterTest, EvalRunnerIT}.java`

**Interfaces:**
- Consumes: `PipelineTrace.queries()` (Task 3); `RetrievalOptions.withRewrite` and `withQueryVariants` (Task 3).
- Produces:
  - `EvalReport.ItemResult` gains a final `List<String> queries`; the 6-argument constructor stays and means none recorded.
  - `EvalConfig.all` returns `vector`, `keyword`, `hybrid`, `hybrid+rewrite`, `hybrid+multiquery` and `hybrid+rewrite+multiquery`. Multi-query uses 3 variants, and every config pins its rewrite and variant settings.
  - The report has a "Queries searched per question" line.

- [ ] **Step 1: Write the failing tests**

In `src/test/java/com/learnings/rag/eval/ReportWriterTest.java`, add:

```java
    @Test
    void reportsHowManyQueriesEachConfigSearchedAndHowOftenExpansionFellBack() {
        ExpectedSource source = new ExpectedSource("a.adoc", "");
        ItemResult expanded = new ItemResult("q01", "Question?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 900,
                List.of(), List.of("Question?", "v1", "v2", "v3"));
        ItemResult fellBack = new ItemResult("q02", "Question 2?", List.of(source), new ItemScore(1, true, 1.0, 1.0), 400,
                List.of(), List.of("Question 2?"));
        RetrievalMetrics.Summary summary = RetrievalMetrics.summarize(List.of(expanded.score(), fellBack.score()),
                List.of(900L, 400L));
        EvalReport report = new EvalReport(Instant.parse("2026-10-06T09:30:00Z"), report(0).run(), List.of(
                new ConfigResult("hybrid+multiquery", new RetrievalOptions(10, 0.0, RetrievalMode.HYBRID, 20, false, 3),
                        summary, List.of(), List.of(expanded, fellBack))));

        assertThat(ReportWriter.markdown(report))
                .contains("Queries searched per question (average): hybrid+multiquery 2.5 (expansion fell back on 1)");
    }
```

In `src/test/java/com/learnings/rag/eval/EvalRunnerIT.java`, change `.containsExactly("vector", "keyword", "hybrid");` to:

```java
                .containsExactly("vector", "keyword", "hybrid", "hybrid+rewrite", "hybrid+multiquery",
                        "hybrid+rewrite+multiquery");
```

Then add this test:

```java
    @Test
    void itemsRecordTheQueriesThatWereSearched() {
        EvalReport report = runner.run(goldenSet(item("q01", new ExpectedSource("pgvector.adoc", ""))),
                EvalConfig.all(properties));

        assertThat(report.configs()).filteredOn(config -> config.name().equals("hybrid"))
                .singleElement().satisfies(config -> assertThat(config.items().getFirst().queries())
                        .containsExactly(HNSW_QUESTION));
        // The stub chat model answers "Stub answer [1].": rewriting searches that text, and expansion falls back to it.
        assertThat(report.configs()).filteredOn(config -> config.name().equals("hybrid+rewrite+multiquery"))
                .singleElement().satisfies(config -> assertThat(config.items().getFirst().queries())
                        .containsExactly("Stub answer [1]."));
    }
```

- [ ] **Step 2: Run them and watch them fail**

Run: `./mvnw -q test -Dtest=ReportWriterTest`
Expected: compilation FAILURE (`ItemResult` has no 7-argument constructor).

- [ ] **Step 3: Implement**

In `src/main/java/com/learnings/rag/eval/EvalReport.java`, replace `ItemResult` with:

```java
    /**
     * @param retrieved the ranked chunks, best first (up to {@link RetrievalMetrics#MRR_K})
     * @param queries the queries actually searched (after rewriting and expansion); empty if not recorded
     */
    public record ItemResult(String id, String question, List<ExpectedSource> expectedSources,
            RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved,
            List<String> queries) {

        public ItemResult(String id, String question, List<ExpectedSource> expectedSources,
                RetrievalMetrics.ItemScore score, long latencyMillis, List<RetrievalMetrics.RankedSource> retrieved) {
            this(id, question, expectedSources, score, latencyMillis, retrieved, List.of());
        }
    }
```

In `src/main/java/com/learnings/rag/eval/EvalRunner.java`, in `run(List<GoldenItem>, EvalConfig)`, replace the `results.add(new ItemResult(…));` statement with:

```java
            List<String> queries = retrieval.trace().queries().isEmpty() ? List.of(item.question())
                    : retrieval.trace().queries();
            results.add(new ItemResult(item.id(), item.question(), item.expectedSources(),
                    RetrievalMetrics.score(item.expectedSources(), ranked), millis, ranked, queries));
```

In `src/main/java/com/learnings/rag/eval/ReportWriter.java`, directly after the loop that writes the summary rows (before the `## By tag` block), insert:

```java
        md.append("\nQueries searched per question (average): ").append(report.configs().stream()
                .map(config -> {
                    double average = config.items().stream().mapToInt(item -> Math.max(1, item.queries().size()))
                            .average().orElse(1);
                    String text = config.name() + " " + String.format(Locale.ROOT, "%.1f", average);
                    if (config.options().queryVariants() > 0) {
                        long fellBack = config.items().stream().filter(item -> item.queries().size() <= 1).count();
                        text += " (expansion fell back on " + fellBack + ")";
                    }
                    return text;
                })
                .collect(joining(" · "))).append('\n');
```

In `src/main/java/com/learnings/rag/eval/EvalConfig.java`, replace `all(...)` and its javadoc with:

```java
    /**
     * The configurations every run compares, all retrieving the top 10 and pinning their rewrite and variant
     * settings. M5 adds rewriting and multi-query (3 variants) on top of hybrid; M6 adds reranking.
     */
    public static List<EvalConfig> all(RagProperties properties) {
        RetrievalOptions base = RetrievalOptions.from(properties).withTopK(RetrievalMetrics.MRR_K)
                .withRewrite(false).withQueryVariants(0);
        RetrievalOptions hybrid = base.withMode(RetrievalMode.HYBRID);
        return List.of(
                new EvalConfig("vector", base.withMode(RetrievalMode.VECTOR)),
                new EvalConfig("keyword", base.withMode(RetrievalMode.KEYWORD)),
                new EvalConfig("hybrid", hybrid),
                new EvalConfig("hybrid+rewrite", hybrid.withRewrite(true)),
                new EvalConfig("hybrid+multiquery", hybrid.withQueryVariants(MULTI_QUERY_VARIANTS)),
                new EvalConfig("hybrid+rewrite+multiquery", hybrid.withRewrite(true).withQueryVariants(MULTI_QUERY_VARIANTS)));
    }

    /** The spec's multi-query: the original question plus 3 variants. */
    static final int MULTI_QUERY_VARIANTS = 3;
```

- [ ] **Step 4: Run them and watch them pass**

Run: `./mvnw -q verify -Dtest=ReportWriterTest -Dit.test=EvalRunnerIT`
Expected: PASS (8 unit tests, 7 ITs).

- [ ] **Step 5: Run every test, then commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add src/main/java/com/learnings/rag/eval src/test/java/com/learnings/rag/eval
git commit -m "feat: eval compares rewrite and multi-query on top of hybrid and reports the queries searched"
```

---

### Task 5: Run the M5 comparison and apply the rule (M5)

**Files:**
- Modify: `src/main/resources/application.yml` only if the rule picks a winner; `eval/README.md`; `README.md`; `docs/superpowers/specs/2026-10-05-rag-pipeline-design.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Check that the utility model setting reaches OpenAI**

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.arguments="--rag.retrieval.utility-model=no-such-model --rag.retrieval.rewrite=true" -Dspring-boot.run.profiles=eval > /tmp/m5-utility-check.log 2>&1; grep -c 'Query rewrite failed' /tmp/m5-utility-check.log`
Expected:
- a count greater than 0, because every rewrite falls back with an error naming `no-such-model`;
- the eval still completes.

This proves that `ChatOptions.model` reaches the OpenAI request and that a failing rewrite degrades instead of failing.

The report from this check is not the M5 result: delete it (`rm` the newest `eval/reports/*`).

- [ ] **Step 2: Run the M5 eval (calls OpenAI: utility model and query embeddings)**

Run it in the background. It makes about 63 × 4 utility calls plus embeddings, which takes several minutes. Keep the Mac awake.

Run: `./mvnw -q spring-boot:run -Dspring-boot.run.profiles=eval`
Expected:
- six summary lines, then `Report written to eval/reports/<time>.md`;
- the report shows the queries-searched line, with `hybrid+multiquery` averaging close to 4.0;
- the By tag table has `conversational`, `identifier` and `untagged` rows.

- [ ] **Step 3: Apply the default rule (spec amendment 32)**

Let N = 63. Compare each candidate (`hybrid+rewrite`, `hybrid+multiquery`, `hybrid+rewrite+multiquery`) with `hybrid` from the **same** report.

A candidate qualifies only if both hold:
- its MRR@10 is ≥ hybrid's + 1/63;
- its hit@5 is ≥ hybrid's − 1/63.

If several qualify, choose the highest MRR@10; on a tie, choose the one with fewer LLM calls (rewrite = 1, multiquery = 1, both = 2).

- **If a winner exists:** set `rewrite` and `query-variants` in `src/main/resources/application.yml` to its values (3 variants for multiquery). Then:
  1. Run `./mvnw -q verify`.
  2. Start the app and ask a conversational question in the browser, for example c01's. The sources must show `join … · found by k queries` for multi-query, or the M4 per-stage scores for rewrite only.
  3. Stop the app.
- **If none qualifies:** leave both off.

- [ ] **Step 4: Record the results**

In `eval/README.md`, add a section `## M5: rewriting and multi-query` directly above `## M4: vector vs keyword vs hybrid` containing:
- the report's run-info table, summary table, queries-searched line and by-tag table, copied verbatim, with the report path;
- one paragraph on the default decision, with the rule's numbers;
- 2–4 bullets on where rewriting and expansion helped or hurt (by tag and by item), what they cost in p50/p95 latency, and whether any expansion fell back.

In `README.md`:
- set the M5 Status row to `✅ done` and M6 to `next`;
- under **Evaluation**, replace the M4 table with the M5 summary table, headed `**M5 comparison (63 questions, <date>):**`, followed by one sentence on the defaults;
- under **Configuration**, add rows for `rag.retrieval.rewrite`, `rag.retrieval.query-variants` and `rag.retrieval.utility-model`;
- in **How answering works** step 1, add one sentence on the optional rewrite and expansion stages.

In the spec's amendment 32, append one sentence on the outcome: which configuration won, if any, with the rule's numbers.

- [ ] **Step 5: Run every test and commit**

Run: `./mvnw -q verify`
Expected: all PASS.

```bash
git add eval/README.md README.md docs/superpowers/specs/2026-10-05-rag-pipeline-design.md src/main/resources/application.yml
git commit -m "eval: M5 comparison of rewriting and multi-query; defaults chosen by the rule"
```

---

## After this plan

M5 is complete when Task 5 Step 4 records the comparison and the defaults.

Next is **M6: the LLM reranker.** `LlmReranker` scores the joined candidates 0–10 with the utility client, using structured output. It keeps the top 5 and drops candidates below a minimum score, so off-topic questions are refused (the spec's E2E check 6, using the sourdough probe). The eval gains `hybrid+rerank` and `full` rows. It gets its own plan.
