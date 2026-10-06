package com.learnings.rag.retrieval;

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
 * Retrieves chunks for a question in the configured {@link RetrievalMode}. HYBRID runs vector and keyword search in
 * parallel on virtual threads and fuses their candidates with reciprocal rank fusion. Every stage is traced. Later
 * milestones add query rewriting, multi-query expansion and reranking here.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final KeywordRetriever keywordRetriever;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, KeywordRetriever keywordRetriever,
            RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.keywordRetriever = keywordRetriever;
        this.defaults = RetrievalOptions.from(properties);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Timed> stages = switch (options.mode()) {
            case VECTOR -> List.of(timed("vector", () -> vectorRetriever.retrieve(new Query(question), options)));
            case KEYWORD -> List.of(timed("keyword", () -> keywordRetriever.retrieve(question, options.topK())));
            case HYBRID -> hybrid(question, options);
        };
        PipelineTrace trace = new PipelineTrace(stages.stream().map(Timed::stage).toList(), millisSince(start));
        return new RetrievalResult(stages.getLast().documents(), trace);
    }

    private List<Timed> hybrid(String question, RetrievalOptions options) {
        int candidates = Math.max(options.candidates(), options.topK());
        Timed vector;
        Timed keyword;
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Timed> vectorSearch = executor.submit(() -> timed("vector",
                    () -> vectorRetriever.retrieve(new Query(question), options.withTopK(candidates))));
            Future<Timed> keywordSearch = executor.submit(() -> timed("keyword",
                    () -> keywordRetriever.retrieve(question, candidates)));
            vector = join(vectorSearch);
            keyword = join(keywordSearch);
        }
        Timed fusion = timed("fusion", () -> ReciprocalRankFusion.fuse(
                List.of(vector.documents(), keyword.documents()), ReciprocalRankFusion.DEFAULT_K, options.topK()));
        return List.of(vector, keyword, fusion);
    }

    /** Waits for a parallel search; its own exception is rethrown as is, not wrapped. */
    private static Timed join(Future<Timed> search) {
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

    private record Timed(String name, List<Document> documents, long millis) {

        PipelineTrace.Stage stage() {
            return new PipelineTrace.Stage(name, millis, documents.stream().map(PipelineTrace.Hit::of).toList());
        }
    }

    private static Timed timed(String name, Supplier<List<Document>> search) {
        long start = System.nanoTime();
        List<Document> documents = search.get();
        return new Timed(name, documents, millisSince(start));
    }

    private static long millisSince(long start) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
    }
}
