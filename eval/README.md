# Evaluation

`golden-set.json` is the reviewed question set that every retrieval change is measured against. The current set was
reviewed by Claude at the user's request, not by a human; see [Baseline](#baseline). Each question
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
- an expected source that matches no indexed chunk: a page that isn't indexed or has no chunks, or a section that
  doesn't exist, for example because `›` was typed as `>`.

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
| recall@5 | Per question, the share of its expected sources found in the top 5, averaged. This is standard IR recall: every listed section counts as a relevant answer location. |
| MRR@10 | Mean of 1 / rank of the first relevant chunk within the top 10 (0 if none). |
| p50 / p95 | Retrieval latency per question in ms, including the query-embedding call (nearest-rank percentiles). |

## Golden set versions

- **v1 (M3, 41 items):** generated, AI-reviewed, plus `h01`–`h04`. The M3 baseline below was measured on v1.
- **v2 (M4, 53 items):** adds `i01`–`i12`. Each is a question naming a Spring AI property the way a developer would
  type it, and each label was checked against the index while planning. These items and `h01` carry
  `"tags": ["identifier"]`, so reports show the identifier questions separately from the rest (`untagged`). Keyword
  search is expected to help most on identifiers.

## M4: vector vs keyword vs hybrid

Recorded on 2026-10-06 from `eval/reports/2026-10-06T05-48-24Z.md` (golden set v2, 53 questions):

| | |
|---|---|
| Golden set | `eval/golden-set.json` (53 questions, sha256 `9d43cfa4e4cb…`) |
| Index | 51 corpus pages, 0 uploads, 1106 chunks |
| Embedding model | `text-embedding-3-small` |
| Chunking | max 500 · min 50 · overlap 60 tokens |

| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| vector | 0.981 | 0.965 | 0.864 | 485 | 777 |
| keyword | 0.509 | 0.472 | 0.359 | 12 | 31 |
| hybrid | 0.962 | 0.937 | 0.736 | 476 | 731 |

| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| vector | identifier | 13 | 1.000 | 0.962 | 0.753 |
| vector | untagged | 40 | 0.975 | 0.967 | 0.900 |
| keyword | identifier | 13 | 0.538 | 0.500 | 0.549 |
| keyword | untagged | 40 | 0.500 | 0.463 | 0.298 |
| hybrid | identifier | 13 | 0.923 | 0.846 | 0.785 |
| hybrid | untagged | 40 | 0.975 | 0.967 | 0.720 |

**Default mode: VECTOR stays.** The rule fixed before the run (spec amendment 24) makes HYBRID the default only if
its MRR@10 is ≥ vector's and its hit@5 is no more than one question (1/53 ≈ 0.019) below vector's. Hybrid's MRR@10 is
0.736 against vector's 0.864, so the rule keeps VECTOR.

What the numbers show:

- **Vector search is strong on this set.** It has 0.981 hit@5, and even identifier questions all land in its top 5
  (it ranks them lower, at MRR@10 0.753, than the rest at 0.900). The golden questions were mostly generated from
  single chunks, which suits vector search.
- **Keyword search is weak because of the ranking function, not the matching.**
  - Postgres tokenizes the identifiers correctly; for example, `spring.ai.vectorstore.neo4j.embedding-dimension`
    matches exactly.
  - But `ts_rank_cd` (cover density) scores an OR query by counting every occurrence of any query term at full
    weight. So a chunk that repeats a common word outranks the chunk holding the rare identifier. In a probe,
    "default" ×10 scored 1.0 while the exact identifier once scored 0.1.
  - Hyphenated identifiers also become phrases (`'…embedding' <-> 'dimens'`), which `ts_rank_cd` scores lower still.
    For `i03` the labelled chunk ranked 82nd of 82 matches.
  - As a result, three different identifier questions got the identical keyword top 3.
  - IDF is *not* the explanation: `ts_rank` has no IDF either and does far better (below).
- **Hybrid makes no measurable difference on identifiers and hurts the rest.**
  - On identifiers its MRR@10 is 0.785 against vector's 0.753. That gap is smaller than one question moving from
    rank 1 to rank 2 (0.038 on 13 items). Its hit@5 is one question worse.
  - On untagged questions its MRR@10 drops from 0.900 to 0.720. Equal-weight RRF lets the noisy `ts_rank_cd` ranking
    push vector's correct rank-1 chunk down.
  - Latency equals vector's: keyword search takes about 12 ms and runs in parallel.
- **The M4 decision applies to the keyword retriever as built (`ts_rank_cd`).** Exploratory results, computed
  outside the harness with an SQL copy of the keyword query, are below. That copy reproduces the harness's
  `ts_rank_cd` numbers exactly; hybrid with these variants was not measured, since that needs query embeddings.

  | Keyword search variant (same OR query and filters) | hit@5 | MRR@10 | identifier hit@5 | identifier MRR@10 |
  |---|---|---|---|---|
  | `ts_rank_cd` (as built) | 0.509 | 0.359 | 0.538 | 0.549 |
  | `ts_rank` | 0.925 | 0.823 | 1.000 | 0.962 |
  | `ts_rank(…, 1)` (length-normalised) | 0.962 | 0.840 | 1.000 | 1.000 |
  | `ts_rank_cd`, all terms first, then any term | 0.604 | 0.484 | 0.846 | 0.857 |

  Switching the rank function to `ts_rank` is a one-line change that could reverse the hybrid result, so it is the
  next experiment. It needs a spec amendment and a re-run of this comparison under the same rule.

## Baseline

Recorded on 2026-10-06 from `eval/reports/2026-10-06T04-51-23Z.md`:

| | |
|---|---|
| Golden set | `eval/golden-set.json` (41 questions, sha256 `d6f40095211d…`) |
| Index | 51 corpus pages, 0 uploads, 1106 chunks |
| Embedding model | `text-embedding-3-small` |
| Chunking | max 500 · min 50 · overlap 60 tokens |

| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| vector | 0.976 | 0.967 | 0.902 | 488 | 830 |

Baseline: vector-only retrieval (M2 pipeline), top 10, text-embedding-3-small. Every later configuration is compared with this row on the same golden set (sha256 above).

About this golden set:

- **Composition:** 41 questions, made up of 37 generated by `gpt-5-mini` from one chunk per page and 4 hand-written ones (`h01`–`h04`).
- **Review:** the review was done by Claude at the user's request, not by a human. Edits were checked against each source excerpt:
  - 3 weak items were dropped;
  - 11 questions were rewritten (for example, one gave away its own answer and another asked two things at once);
  - 3 items gained a second source.

  A human pass may still move the numbers.
- **Labels corrected after the whole-branch review:** for `q13`, `q16` and `q38`, a section ranked above the label
  answers the question just as well. Examples are the identical "Auto-pulling Models" section on the Ollama chat page,
  and the `tools.adoc` section on exposing beans with `@McpTool`. These sections were added as alternative sources,
  which raised MRR@10 from 0.862 to 0.902. Six other items had higher-ranked chunks that do *not* answer (link lists,
  intros), and they were left unchanged.
- **The one miss** is `q32` (Neo4j prerequisites). The right page is retrieved, but its Prerequisites section is not in the top 10. It is kept as a genuine retrieval miss.
- **Near-ceiling numbers:** they are expected. A question written from one chunk is semantically close to that chunk, which favours vector search. The headroom for M4–M6 is mostly in MRR@10 and in harder questions: identifier lookups, troubleshooting phrasing, and answers that span several sections. Add such items before M4 to make its comparison sharper.
