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
