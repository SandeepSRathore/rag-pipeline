package com.learnings.rag.ingest;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

import com.learnings.rag.ingest.DocumentIngestionService.IngestOutcome;

/** Makes the index match a directory: new and changed files are ingested, deleted files are removed. */
@Service
public class CorpusIngestor {

    private static final Logger log = LoggerFactory.getLogger(CorpusIngestor.class);

    private final DocumentIngestionService ingestion;
    private final SourceDocumentRepository documents;

    public CorpusIngestor(DocumentIngestionService ingestion, SourceDocumentRepository documents) {
        this.ingestion = ingestion;
        this.documents = documents;
    }

    public record CorpusReport(int added, int updated, int skipped, int removed, int chunksWritten) {
    }

    public CorpusReport ingestDirectory(Path directory) throws IOException {
        if (!Files.isDirectory(directory)) {
            throw new CorpusUnavailableException("Corpus directory " + directory.toAbsolutePath()
                    + " does not exist. Run scripts/fetch-corpus.sh first.");
        }
        List<Path> files;
        try (Stream<Path> walk = Files.walk(directory)) {
            files = walk.filter(Files::isRegularFile).filter(CorpusIngestor::isSupported).sorted().toList();
        }
        if (files.isEmpty()) {
            throw new CorpusUnavailableException("Corpus directory " + directory.toAbsolutePath()
                    + " contains no .adoc or .md files. Run scripts/fetch-corpus.sh first.");
        }

        int added = 0;
        int updated = 0;
        int skipped = 0;
        int chunksWritten = 0;
        Set<String> seen = new HashSet<>();
        for (Path file : files) {
            String sourcePath = directory.relativize(file).toString().replace(File.separatorChar, '/');
            seen.add(sourcePath);
            IngestOutcome outcome = ingestion.ingest(sourcePath, baseName(file), Files.readString(file),
                    SourceDocument.Origin.CORPUS);
            chunksWritten += outcome.chunksWritten();
            switch (outcome.status()) {
                case ADDED -> added++;
                case UPDATED -> updated++;
                case SKIPPED -> skipped++;
            }
            log.debug("{} {} ({} chunks)", outcome.status(), sourcePath, outcome.document().chunkCount());
        }

        int removed = 0;
        for (SourceDocument document : documents.findByOrigin(SourceDocument.Origin.CORPUS)) {
            if (!seen.contains(document.sourcePath()) && ingestion.delete(document.id())) {
                removed++;
            }
        }

        CorpusReport report = new CorpusReport(added, updated, skipped, removed, chunksWritten);
        log.info("Corpus ingest: {}", report);
        return report;
    }

    private static boolean isSupported(Path file) {
        String name = file.getFileName().toString();
        return name.endsWith(".adoc") || name.endsWith(".md");
    }

    private static String baseName(Path file) {
        String name = file.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }
}
