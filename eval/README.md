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
