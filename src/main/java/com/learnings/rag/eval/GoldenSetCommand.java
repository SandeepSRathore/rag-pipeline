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
