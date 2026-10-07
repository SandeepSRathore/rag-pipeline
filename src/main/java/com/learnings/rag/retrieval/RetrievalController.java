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
