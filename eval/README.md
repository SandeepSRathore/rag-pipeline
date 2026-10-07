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
- no `expectedSources`, unless it is tagged `unanswerable`. An unanswerable item must have `"expectedSources": []`,
  and you should grep the corpus to confirm that nothing answers it;
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
| refused: unanswerable | Unanswerable questions whose retrieval came back empty, out of all unanswerable ones. Chat then answers "I couldn't find…" without calling the model. Higher is better. |
| refused: answerable | Answerable questions whose retrieval came back empty (false refusals). They also count as misses in hit@5. |
| p50 / p95 | Retrieval latency per answerable question in ms, including the query-embedding call (nearest-rank percentiles). |

hit@5, recall@5, MRR@10 and latency cover only the answerable questions.

## Golden set versions

- **v1 (M3, 41 items):** generated, AI-reviewed, plus `h01`–`h04`. The M3 baseline below was measured on v1.
- **v2 (M4, 53 items):** adds `i01`–`i12`. Each is a question naming a Spring AI property the way a developer would
  type it, and each label was checked against the index while planning. These items and `h01` carry
  `"tags": ["identifier"]`, so reports show the identifier questions separately from the rest (`untagged`). Keyword
  search is expected to help most on identifiers.
- **v3 (M5, 63 items):** adds `c01`–`c10`. Each is a chatty, vague rephrasing of a verified item (typos, filler,
  symptoms instead of terms), with that item's labels and `"tags": ["conversational"]`. They are written by Claude to
  test query rewriting and multi-query expansion, which target exactly this kind of question.
- **v4 (M6, 73 items):** adds `u01`–`u10`, questions the corpus does **not** answer, with `"expectedSources": []` and
  `"tags": ["unanswerable"]`. Three are off-topic (sourdough, running, carpet stains). Seven are near the domain:
  providers, stores and Spring projects that the 52 pages don't cover. They were written by Claude, and a grep confirmed
  the corpus doesn't answer them. They measure refusals; hit@5, recall@5, MRR@10 and latency still cover the 63
  answerable questions, so the numbers stay comparable with v3.

## M6: reranking and refusals

Recorded on 2026-10-07 from `eval/reports/2026-10-07T03-58-23Z.md` (run 1; golden set v4: 63 answerable and 10
unanswerable questions; reranker `gpt-4.1-mini` at temperature 0):

| | |
|---|---|
| Golden set | `eval/golden-set.json` (73 questions, sha256 `8eda988add77…`) |
| Index | 51 corpus pages, 0 uploads, 1106 chunks |
| Embedding model | `text-embedding-3-small` |
| Chunking | max 500 · min 50 · overlap 60 tokens |

| Config | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| vector | 0.984 | 0.971 | 0.851 | 0/10 | 0 | 486 | 1059 |
| keyword | 0.952 | 0.942 | 0.830 | 0/10 | 0 | 17 | 44 |
| hybrid | 0.984 | 0.979 | 0.905 | 0/10 | 0 | 467 | 561 |
| hybrid+multiquery | 0.968 | 0.963 | 0.829 | 0/10 | 0 | 1860 | 2346 |
| hybrid+rerank | 0.984 | 0.984 | 0.976 | 0/10 | 0 | 3782 | 4805 |
| hybrid+multiquery+rerank | 0.984 | 0.979 | 0.937 | 0/10 | 0 | 5082 | 5768 |

Answerable questions: 63; hit@5, recall@5, MRR@10 and latency cover these. *Refused* = retrieval came back empty, so chat answers "I couldn't find…" without calling the model.

The rerank rows ran at min-score 0, so they refuse nothing. The sweeps below show what each minimum would have
scored.

Queries searched per question (average): vector 1.0 · keyword 1.0 · hybrid 1.0 · hybrid+multiquery 4.0 (expansion fell back on 0) · hybrid+rerank 1.0 · hybrid+multiquery+rerank 4.0 (expansion fell back on 0)

Rerank fell back to the fused order (questions): hybrid+rerank 0 · hybrid+multiquery+rerank 0

| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| vector | conversational | 10 | 1.000 | 1.000 | 0.783 |
| vector | identifier | 13 | 1.000 | 0.962 | 0.753 |
| vector | untagged | 40 | 0.975 | 0.967 | 0.900 |
| keyword | conversational | 10 | 0.900 | 0.900 | 0.781 |
| keyword | identifier | 13 | 1.000 | 1.000 | 1.000 |
| keyword | untagged | 40 | 0.950 | 0.933 | 0.788 |
| hybrid | conversational | 10 | 1.000 | 1.000 | 0.850 |
| hybrid | identifier | 13 | 1.000 | 1.000 | 0.962 |
| hybrid | untagged | 40 | 0.975 | 0.967 | 0.900 |
| hybrid+multiquery | conversational | 10 | 1.000 | 1.000 | 0.733 |
| hybrid+multiquery | identifier | 13 | 1.000 | 1.000 | 0.904 |
| hybrid+multiquery | untagged | 40 | 0.950 | 0.942 | 0.829 |
| hybrid+rerank | conversational | 10 | 1.000 | 1.000 | 0.950 |
| hybrid+rerank | identifier | 13 | 1.000 | 1.000 | 1.000 |
| hybrid+rerank | untagged | 40 | 0.975 | 0.975 | 0.975 |
| hybrid+multiquery+rerank | conversational | 10 | 1.000 | 1.000 | 1.000 |
| hybrid+multiquery+rerank | identifier | 13 | 1.000 | 1.000 | 0.923 |
| hybrid+multiquery+rerank | untagged | 40 | 0.975 | 0.967 | 0.925 |

**`hybrid+rerank`: min-score sweep** (run 1). Each row is what the run would have scored with
`rag.retrieval.rerank.min-score` at that value:

| min-score | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable |
|---|---|---|---|---|---|
| 0 | 0.984 | 0.984 | 0.976 | 0/10 | 0 |
| 1 | 0.984 | 0.984 | 0.976 | 4/10 | 0 |
| 2 | 0.984 | 0.984 | 0.976 | 5/10 | 0 |
| 3 | 0.984 | 0.984 | 0.976 | 5/10 | 0 |
| 4 | 0.984 | 0.984 | 0.976 | 6/10 | 0 |
| 5 | 0.984 | 0.984 | 0.976 | 6/10 | 0 |
| 6 | 0.984 | 0.984 | 0.976 | 8/10 | 0 |
| 7 | 0.984 | 0.968 | 0.976 | 8/10 | 0 |
| 8 | 0.968 | 0.937 | 0.960 | 8/10 | 1 |
| 9 | 0.952 | 0.902 | 0.944 | 8/10 | 2 |
| 10 | 0.746 | 0.688 | 0.746 | 10/10 | 14 |

**`hybrid+multiquery+rerank`: min-score sweep** (run 1):

| min-score | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable |
|---|---|---|---|---|---|
| 0 | 0.984 | 0.979 | 0.937 | 0/10 | 0 |
| 1 | 0.984 | 0.979 | 0.937 | 5/10 | 0 |
| 2 | 0.984 | 0.979 | 0.937 | 5/10 | 0 |
| 3 | 0.984 | 0.979 | 0.937 | 5/10 | 0 |
| 4 | 0.984 | 0.979 | 0.937 | 6/10 | 0 |
| 5 | 0.984 | 0.979 | 0.937 | 6/10 | 0 |
| 6 | 0.984 | 0.971 | 0.937 | 7/10 | 0 |
| 7 | 0.984 | 0.963 | 0.937 | 8/10 | 0 |
| 8 | 0.984 | 0.955 | 0.937 | 8/10 | 0 |
| 9 | 0.937 | 0.894 | 0.897 | 8/10 | 2 |
| 10 | 0.698 | 0.656 | 0.683 | 9/10 | 16 |

**Run 2, the stability check** (`eval/reports/2026-10-07T04-13-41Z.md`), summary rows at min-score 0:

| Config | hit@5 | recall@5 | MRR@10 | refused: unanswerable | refused: answerable | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| hybrid | 0.984 | 0.979 | 0.905 | 0/10 | 0 | 457 | 557 |
| hybrid+rerank | 0.984 | 0.984 | 0.952 | 0/10 | 0 | 3578 | 4185 |
| hybrid+multiquery+rerank | 0.984 | 0.979 | 0.944 | 0/10 | 0 | 5291 | 6424 |

In run 2, `hybrid+rerank` at min-score 6 scores hit@5 0.984, recall@5 0.984 and MRR@10 0.952, and refuses 8/10
unanswerable questions and 0 answerable ones.

**Defaults: reranking is on, with min-score 6.** The rules fixed before the run (spec amendment 37) chose
`hybrid+rerank` at min-score 6:
- **Rule T (min-score):** in run 1, min-scores 0–7 keep hit@5 and recall@5 within 1/63 of min-score 0. At 8, recall@5
  falls to 0.937 and an answerable question is refused. Both 6 and 7 refuse 8 of 10 unanswerable questions, so the
  lower one, 6, wins. For `hybrid+multiquery+rerank`, rule T picks 7.
- **Rule D (default):** at min-score 6, hit@5 0.984 equals hybrid's, and MRR@10 0.976 beats hybrid's 0.905 by 0.071,
  more than four questions' worth (1/63 ≈ 0.016). It also refuses 8 of 10. Both rerank rows qualify, and
  `hybrid+multiquery+rerank` (MRR@10 0.937) doesn't beat `hybrid+rerank` by 1/63, so the cheaper one wins.
- **Stability:** run 2, on its own, picks the same configuration and the same min-score. At 6 it qualifies again:
  MRR@10 0.952 against hybrid's 0.905, 8 of 10 refused, no false refusals.

What the numbers show:

- **Reranking fixed the order, not the coverage.**
  - In run 1 it moved 8 answerable questions up and none down: `q10` and `h03` from rank 4 to 1, and six others from
    rank 2 to 1. In run 2, 6 moved up and one (`c03`) moved down, from rank 1 to 2.
  - In run 1, MRR@10 rose on every tag: conversational 0.850 → 0.950, identifier 0.962 → 1.000, untagged
    0.900 → 0.975. In run 2, conversational reached 0.900 and untagged 0.963, while identifier stayed at 0.962.
  - hit@5 stays at 0.984. The one miss is still `q32`: its top three chunks come from the right page (Neo4j, rated
    9–10), but none of the top 10 from the expected section.
- **Refusals: 8 of 10 at min-score 6, with no false refusals in either run.**
  - The off-topic questions (`u01`–`u03`) and the Hibernate one (`u09`) get top ratings of 0. Pinecone (`u06`) gets 1,
    the cron job (`u10`) 3, and fine-tuning (`u07`) and Spring Batch (`u08`) 5.
  - Two near-domain questions get through. watsonx (`u04`) is matched to OpenAI's chat properties, and Couchbase
    (`u05`) to PGvector's auto-configuration, both rated 8–9. The judge treats the same kind of answer for another
    product as an answer. For those, the answer prompt's own "couldn't find" rule is the remaining guard; M7 measures
    it.
  - **The margin is narrow.** The lowest top rating of an answerable question is 7 (`i03`, in both runs), while
    refused near-domain questions reach 5. From 8 up, answerable questions start being refused.
  - **These refusal numbers are in-sample.** Min-score 6 was chosen on these same questions, and run 2 asked them again.
    On new questions, expect some false refusals and more unanswerable questions getting through.
- **The cost is about 3 s per question.**
  - p50 retrieval goes from 0.47 s to 3.8 s (3.6 s in run 2). One `gpt-4.1-mini` call reads about 4,000 tokens of
    passages and writes 20 ratings.
  - Chat now shows its sources after about 4 s instead of 0.5 s.
  - A refused question costs the same retrieval time but no answer-model call.
- **Reranking partly rescues multi-query.** It lifts MRR@10 from 0.829 to 0.937 (0.850 to 0.944 in run 2), but stays
  below rerank alone and costs another 1.3–1.7 s. Multi-query stays off.
- **Run-to-run variance is visible.**
  - `hybrid+rerank`'s MRR@10 was 0.976 in run 1 and 0.952 in run 2.
  - Multi-query alone scored 0.829 and 0.850, and 0.842 in M5.
  - The decision held in both runs.

## M5: rewriting and multi-query

Recorded on 2026-10-06 from `eval/reports/2026-10-06T10-56-23Z.md` (golden set v3, 63 questions; utility model `gpt-4.1-mini` at temperature 0):

| | |
|---|---|
| Golden set | `eval/golden-set.json` (63 questions, sha256 `0cd74f160e10…`) |
| Index | 51 corpus pages, 0 uploads, 1106 chunks |
| Embedding model | `text-embedding-3-small` |
| Chunking | max 500 · min 50 · overlap 60 tokens |

| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| vector | 0.984 | 0.971 | 0.851 | 500 | 658 |
| keyword | 0.952 | 0.942 | 0.830 | 18 | 40 |
| hybrid | 0.984 | 0.979 | 0.905 | 492 | 673 |
| hybrid+rewrite | 0.921 | 0.915 | 0.784 | 1464 | 1744 |
| hybrid+multiquery | 0.968 | 0.963 | 0.842 | 2032 | 2325 |
| hybrid+rewrite+multiquery | 0.952 | 0.947 | 0.761 | 2982 | 3289 |

Queries searched per question (average): vector 1.0 · keyword 1.0 · hybrid 1.0 · hybrid+rewrite 1.0 · hybrid+multiquery 4.0 (expansion fell back on 0) · hybrid+rewrite+multiquery 4.0 (expansion fell back on 0)

| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| vector | conversational | 10 | 1.000 | 1.000 | 0.783 |
| vector | identifier | 13 | 1.000 | 0.962 | 0.753 |
| vector | untagged | 40 | 0.975 | 0.967 | 0.900 |
| keyword | conversational | 10 | 0.900 | 0.900 | 0.781 |
| keyword | identifier | 13 | 1.000 | 1.000 | 1.000 |
| keyword | untagged | 40 | 0.950 | 0.933 | 0.788 |
| hybrid | conversational | 10 | 1.000 | 1.000 | 0.850 |
| hybrid | identifier | 13 | 1.000 | 1.000 | 0.962 |
| hybrid | untagged | 40 | 0.975 | 0.967 | 0.900 |
| hybrid+rewrite | conversational | 10 | 1.000 | 1.000 | 0.833 |
| hybrid+rewrite | identifier | 13 | 1.000 | 1.000 | 0.865 |
| hybrid+rewrite | untagged | 40 | 0.875 | 0.867 | 0.745 |
| hybrid+multiquery | conversational | 10 | 1.000 | 1.000 | 0.783 |
| hybrid+multiquery | identifier | 13 | 1.000 | 1.000 | 0.904 |
| hybrid+multiquery | untagged | 40 | 0.950 | 0.942 | 0.837 |
| hybrid+rewrite+multiquery | conversational | 10 | 1.000 | 1.000 | 0.758 |
| hybrid+rewrite+multiquery | identifier | 13 | 1.000 | 1.000 | 0.910 |
| hybrid+rewrite+multiquery | untagged | 40 | 0.925 | 0.917 | 0.713 |

**Defaults: rewriting and multi-query stay off.** The rule fixed before the run (spec amendment 32) needs a candidate to
beat plain hybrid's MRR@10 (0.905) by at least 1/63 without losing more than one question on hit@5 (0.984). All three
candidates scored *below* hybrid on both:
- `hybrid+rewrite`: MRR@10 0.784, hit@5 0.921;
- `hybrid+multiquery`: 0.842 / 0.968;
- `hybrid+rewrite+multiquery`: 0.761 / 0.952.

What the numbers show (single run, so per-question anecdotes are single samples):

- **Rewriting tended to pad queries with "Spring AI".**
  - `gpt-4.1-mini` added "…in Spring AI documentation" or "…in Spring AI" to 34 of 63 questions. That is likely an
    artifact of the target-system phrase given to `RewriteQueryTransformer`.
  - 13 of the 15 questions that got worse had "Spring" added. For example, `q28` fell from rank 1 to 8, `q24` to 6,
    and `q17` out of the top 10. Untagged MRR@10 fell from 0.900 to 0.745.
  - But 21 padded questions did not get worse, so this is an association, not a measured mechanism.
  - The added terms are broad rather than universal: "spring" or "ai" occurs in 487 of 1,106 chunks (44%), and
    "document" in 184 (17%).
  - Rewriting helped some chatty questions (`c01` and `c06` rose to rank 1), but the conversational tag as a whole
    stayed flat (0.833 against 0.850).
- **Multi-query variants were padded the same way, by our own prompt.**
  - 83 of 189 variants (44%) add "Spring…" that the question lacks, 32 of them "Spring Boot". The expansion prompt
    itself names "Spring AI reference documentation (Java, Spring Boot)".
  - Variants that pull towards generic pages produce rankings that disagree with the original question's precise one,
    and fusing them diluted its top hit: MRR@10 0.842 against 0.905, and conversational 0.783 against 0.850.
  - Expansion itself worked: 4.0 queries per question, 0 fallbacks.
- **`hybrid+rewrite+multiquery` never searched the user's question.** Expansion runs on the rewritten text (spec
  amendment 30), so this configuration searched the rewrite plus 3 variants of it, on all 63 items. Its result says
  nothing about adding variants to the original question.
- **The cost is real.** p50 retrieval latency goes from 0.49 s (hybrid) to 1.46 s with rewriting, 2.03 s with
  multi-query and 2.98 s with both.
- **Caveats:**
  - This is a single run. Temperature 0 is not fully deterministic: the same question was rewritten differently by
    the two rewrite configurations in 14 of 63 cases.
  - The decision itself is robust: multi-query would have needed +0.079 MRR@10 to qualify.
  - The golden questions are mostly clean, single-intent questions, and hybrid already reaches hit@5 1.000 on the
    conversational ones, which leaves these techniques little room.
  - With 63 questions, one question is worth about 0.016 in hit@5.
- **Next ideas (not measured):**
  - Remove product names ("Spring AI", "Spring Boot") from the rewrite target phrase and from the expansion prompt.
  - Rewrite only questions that look conversational.
  - Search the original question alongside the rewrite.
  - Repeat each LLM configuration 2–3 times to show the variance.

## M4: vector vs keyword vs hybrid

**Final comparison** from `eval/reports/2026-10-06T09-34-14Z.md` (golden set v2, 53 questions). Keyword search here is ranked with
length-normalised `ts_rank` (spec amendment 27):

| | |
|---|---|
| Golden set | `eval/golden-set.json` (53 questions, sha256 `9d43cfa4e4cb…`) |
| Index | 51 corpus pages, 0 uploads, 1106 chunks |
| Embedding model | `text-embedding-3-small` |
| Chunking | max 500 · min 50 · overlap 60 tokens |

| Config | hit@5 | recall@5 | MRR@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| vector | 0.981 | 0.965 | 0.864 | 497 | 885 |
| keyword | 0.962 | 0.950 | 0.840 | 16 | 52 |
| hybrid | 0.981 | 0.975 | 0.915 | 469 | 715 |

| Config | Tag | Items | hit@5 | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| vector | identifier | 13 | 1.000 | 0.962 | 0.753 |
| vector | untagged | 40 | 0.975 | 0.967 | 0.900 |
| keyword | identifier | 13 | 1.000 | 1.000 | 1.000 |
| keyword | untagged | 40 | 0.950 | 0.933 | 0.788 |
| hybrid | identifier | 13 | 1.000 | 1.000 | 0.962 |
| hybrid | untagged | 40 | 0.975 | 0.967 | 0.900 |

**Default mode: HYBRID.** The rule fixed before the first run (spec amendment 24) holds:
- hybrid's MRR@10 is 0.915, at least vector's 0.864;
- hybrid's hit@5 is 0.981, equal to vector's (the rule allows one question, 1/53, below).

`rag.retrieval.mode` is now `hybrid`.

- **The whole gain is on identifier questions.** MRR@10 rises from 0.753 (vector) to 0.962 (hybrid); keyword search
  alone ranks every identifier question's section first (1.000).
- **Nothing is lost on the other 40 questions.** Hybrid and vector both score hit@5 0.975 and MRR@10 0.900.
- **Latency is unchanged.** Keyword search takes about 16 ms and runs in parallel with the query embedding.
- **Caveats:**
  - `ts_rank` normalization 1 was chosen after seeing exploratory numbers on this same golden set, which risks
    overfitting to it.
  - The identifier questions were written by Claude.
  - With 53 questions, one question is worth about 0.019 in hit@5.

### First run: keyword ranked by `ts_rank_cd`

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

**Default after the first run: VECTOR stayed.** The rule fixed before the run (spec amendment 24) makes HYBRID the default only if
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

  Switching the rank function to `ts_rank` was a one-line change that could reverse the hybrid result. It was done
  (spec amendment 27), and the comparison was re-run under the same rule; that run is the final comparison above.

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

Baseline: vector-only retrieval (M2 pipeline), top 10, text-embedding-3-small. Later configurations were compared with this row on golden set v1; from M4 on, configurations are compared on v2 within one run (see M4 above).

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
