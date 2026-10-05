package com.learnings.rag.ingest;

import java.io.IOException;
import java.util.List;
import java.util.UUID;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ProblemDetail;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.learnings.rag.config.RagProperties;
import com.learnings.rag.ingest.CorpusIngestor.CorpusReport;
import com.learnings.rag.ingest.DocumentIngestionService.IngestOutcome;

@RestController
@RequestMapping("/api")
public class DocumentController {

    private final CorpusIngestor corpusIngestor;
    private final DocumentIngestionService ingestion;
    private final SourceDocumentRepository documents;
    private final TextExtractor textExtractor;
    private final RagProperties properties;

    public DocumentController(CorpusIngestor corpusIngestor, DocumentIngestionService ingestion,
            SourceDocumentRepository documents, TextExtractor textExtractor, RagProperties properties) {
        this.corpusIngestor = corpusIngestor;
        this.ingestion = ingestion;
        this.documents = documents;
        this.textExtractor = textExtractor;
        this.properties = properties;
    }

    @PostMapping("/ingest/corpus")
    public CorpusReport ingestCorpus() throws IOException {
        return corpusIngestor.ingestDirectory(properties.corpusDir());
    }

    @GetMapping("/documents")
    public List<SourceDocument> list() {
        return documents.findAll();
    }

    @PostMapping(path = "/documents", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<IngestOutcome> upload(@RequestParam("file") MultipartFile file) throws IOException {
        String filename = baseFilename(file.getOriginalFilename());
        if (!textExtractor.supports(filename)) {
            throw new ResponseStatusException(HttpStatus.UNSUPPORTED_MEDIA_TYPE,
                    "Supported types: .adoc .asciidoc .md .markdown .txt .pdf .html .htm .docx");
        }
        String text;
        try {
            text = textExtractor.extract(filename, file.getBytes());
        }
        catch (RuntimeException e) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_CONTENT, "Could not read " + filename, e);
        }
        if (text.isBlank()) {
            throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_CONTENT,
                    "No extractable text in " + filename + " (scanned PDFs need OCR first)");
        }
        IngestOutcome outcome = ingestion.ingest("uploads/" + filename, stripExtension(filename), text,
                SourceDocument.Origin.UPLOAD);
        HttpStatus status = outcome.status() == IngestOutcome.Status.ADDED ? HttpStatus.CREATED : HttpStatus.OK;
        return ResponseEntity.status(status).body(outcome);
    }

    @DeleteMapping("/documents/{id}")
    public ResponseEntity<Void> delete(@PathVariable UUID id) {
        return ingestion.delete(id) ? ResponseEntity.noContent().build() : ResponseEntity.notFound().build();
    }

    @ExceptionHandler(CorpusUnavailableException.class)
    ProblemDetail corpusUnavailable(CorpusUnavailableException e) {
        return ProblemDetail.forStatusAndDetail(HttpStatus.CONFLICT, e.getMessage());
    }

    /** Keeps only the last path segment, so "../../etc/x.md" or "C:\docs\x.md" becomes "x.md". */
    static String baseFilename(String original) {
        String name = original == null ? "" : original.substring(Math.max(original.lastIndexOf('/'), original.lastIndexOf('\\')) + 1);
        if (name.isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "The uploaded file needs a name");
        }
        return name;
    }

    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}
