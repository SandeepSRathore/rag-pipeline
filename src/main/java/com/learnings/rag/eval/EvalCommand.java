package com.learnings.rag.eval;

import java.nio.file.Path;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/**
 * {@code ./mvnw spring-boot:run -Dspring-boot.run.profiles=eval}: scores retrieval (and with
 * {@code --rag.eval.generation=true} the answers) against the golden set.
 */
@Component
@Profile("eval")
public class EvalCommand implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(EvalCommand.class);

    private final GoldenSetFile files;
    private final EvalRunner runner;
    private final GenerationEvalRunner generationRunner;
    private final ReportWriter reportWriter;
    private final EvalProperties properties;
    private final RagProperties ragProperties;

    public EvalCommand(GoldenSetFile files, EvalRunner runner, GenerationEvalRunner generationRunner,
            ReportWriter reportWriter, EvalProperties properties,
            RagProperties ragProperties) {
        this.files = files;
        this.runner = runner;
        this.generationRunner = generationRunner;
        this.reportWriter = reportWriter;
        this.properties = properties;
        this.ragProperties = ragProperties;
    }

    @Override
    public void run(ApplicationArguments args) {
        GoldenSet goldenSet = files.read(properties.goldenSet());
        log.info("Evaluating {} questions from {}", goldenSet.items().size(), goldenSet.path());
        EvalReport report = runner.run(goldenSet, EvalConfig.all(ragProperties)); // logs one summary line per config
        if (properties.generation()) {
            log.info("Generating and judging an answer for each of the {} questions", goldenSet.items().size());
            report = report.withGeneration(generationRunner.run(goldenSet.items()));
        }
        Path markdown = reportWriter.write(report, properties.reportsDir());
        log.info("Report written to {}", markdown);
    }
}
