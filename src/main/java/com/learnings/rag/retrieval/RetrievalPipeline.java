package com.learnings.rag.retrieval;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
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
 * Retrieves chunks for a question:
 * <ol>
 * <li>optionally rewrites it;</li>
 * <li>optionally expands it into the original plus variants;</li>
 * <li>searches every query in the configured {@link RetrievalMode}, in parallel on virtual threads;</li>
 * <li>joins several rankings with reciprocal rank fusion;</li>
 * <li>optionally reranks the candidates with {@link LlmReranker}, drops those rated below the minimum score and keeps
 * the best top K.</li>
 * </ol>
 * HYBRID runs vector and keyword search in parallel for each query. Every stage is traced; a single query keeps
 * M4's stage names ({@code vector}, {@code keyword}, {@code fusion}), several get a {@code q1:} prefix and a final
 * {@code join}; reranking adds {@code rerank}, or {@link PipelineTrace#RERANK_FAILED} when it failed.
 */
@Service
public class RetrievalPipeline {

    private final VectorRetriever vectorRetriever;
    private final KeywordRetriever keywordRetriever;
    private final QueryRewriter queryRewriter;
    private final LlmQueryExpander queryExpander;
    private final LlmReranker reranker;
    private final RetrievalOptions defaults;

    public RetrievalPipeline(VectorRetriever vectorRetriever, KeywordRetriever keywordRetriever,
            QueryRewriter queryRewriter, LlmQueryExpander queryExpander, LlmReranker reranker,
            RagProperties properties) {
        this.vectorRetriever = vectorRetriever;
        this.keywordRetriever = keywordRetriever;
        this.queryRewriter = queryRewriter;
        this.queryExpander = queryExpander;
        this.reranker = reranker;
        this.defaults = RetrievalOptions.from(properties);
    }

    public RetrievalResult retrieve(String question, RetrievalOptions options) {
        long start = System.nanoTime();
        List<Timed> stages = new ArrayList<>();

        String searchText = question;
        if (options.rewrite()) {
            Timed rewrite = timedQueries("rewrite", () -> List.of(queryRewriter.rewrite(question)));
            stages.add(rewrite);
            searchText = rewrite.queries().getFirst();
        }
        List<String> queries = List.of(searchText);
        if (options.queryVariants() > 0) {
            String text = searchText;
            Timed expand = timedQueries("expand", () -> queryExpander.expand(new Query(text), options.queryVariants())
                    .stream().map(Query::text).toList());
            stages.add(expand);
            queries = expand.queries();
        }

        // The reranker rates `candidates` chunks and keeps the best topK; without it, the search keeps topK directly.
        int depth = options.rerank() ? Math.max(options.candidates(), options.topK()) : options.topK();
        List<Document> documents;
        if (queries.size() == 1) {
            List<Timed> search = search(queries.getFirst(), options, depth, "");
            stages.addAll(search);
            documents = search.getLast().documents();
        }
        else {
            List<List<Timed>> perQuery = searchInParallel(queries, options);
            perQuery.forEach(stages::addAll);
            Timed join = timed("join", () -> ReciprocalRankFusion.fuse(
                    perQuery.stream().map(search -> search.getLast().documents()).toList(),
                    ReciprocalRankFusion.DEFAULT_K, depth));
            stages.add(join);
            documents = join.documents();
        }
        if (options.rerank()) {
            documents = rerank(question, documents, options, stages);
        }
        PipelineTrace trace = new PipelineTrace(stages.stream().map(Timed::stage).toList(), millisSince(start));
        return new RetrievalResult(documents, trace);
    }

    /** Retrieves with the configured defaults ({@code rag.retrieval.*}). */
    public RetrievalResult retrieve(String question) {
        return retrieve(question, defaults);
    }

    /**
     * Rates the candidates against the user's own question (not a rewrite), drops those rated below the minimum and
     * keeps the best topK. If rating fails, the fused order is kept and no minimum applies: a reranker outage must not
     * turn every question into a refusal.
     */
    private List<Document> rerank(String question, List<Document> candidates, RetrievalOptions options,
            List<Timed> stages) {
        long start = System.nanoTime();
        Optional<List<Document>> rated = reranker.rerank(question, candidates);
        if (rated.isEmpty()) {
            stages.add(new Timed(PipelineTrace.RERANK_FAILED, List.of(), List.of(), millisSince(start)));
            return candidates.stream().limit(options.topK()).toList();
        }
        stages.add(new Timed("rerank", rated.get(), List.of(), millisSince(start)));
        return rated.get().stream()
                .filter(document -> document.getScore() >= options.minScore())
                .limit(options.topK())
                .toList();
    }

    /** Every query in parallel; each keeps {@code candidates} chunks so the join has depth to fuse. */
    private List<List<Timed>> searchInParallel(List<String> queries, RetrievalOptions options) {
        int depth = Math.max(options.candidates(), options.topK());
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            List<Future<List<Timed>>> searches = new ArrayList<>();
            for (int i = 0; i < queries.size(); i++) {
                String query = queries.get(i);
                String prefix = "q" + (i + 1) + ":";
                searches.add(executor.submit(() -> search(query, options, depth, prefix)));
            }
            List<List<Timed>> results = new ArrayList<>();
            for (Future<List<Timed>> search : searches) {
                results.add(join(search));
            }
            return results;
        }
    }

    /** One query in the configured mode; the last stage holds its final ranking of at most {@code limit} chunks. */
    private List<Timed> search(String query, RetrievalOptions options, int limit, String prefix) {
        return switch (options.mode()) {
            case VECTOR -> List.of(timed(prefix + "vector",
                    () -> vectorRetriever.retrieve(new Query(query), options.withTopK(limit))));
            case KEYWORD -> List.of(timed(prefix + "keyword", () -> keywordRetriever.retrieve(query, limit)));
            case HYBRID -> hybrid(query, options, limit, prefix);
        };
    }

    private List<Timed> hybrid(String query, RetrievalOptions options, int limit, String prefix) {
        int candidates = Math.max(options.candidates(), limit);
        Timed vector;
        Timed keyword;
        try (ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor()) {
            Future<Timed> vectorSearch = executor.submit(() -> timed(prefix + "vector",
                    () -> vectorRetriever.retrieve(new Query(query), options.withTopK(candidates))));
            Future<Timed> keywordSearch = executor.submit(() -> timed(prefix + "keyword",
                    () -> keywordRetriever.retrieve(query, candidates)));
            vector = join(vectorSearch);
            keyword = join(keywordSearch);
        }
        Timed fusion = timed(prefix + "fusion", () -> ReciprocalRankFusion.fuse(
                List.of(vector.documents(), keyword.documents()), ReciprocalRankFusion.DEFAULT_K, limit));
        return List.of(vector, keyword, fusion);
    }

    /** Waits for a parallel search; its own exception is rethrown as is, not wrapped. */
    private static <T> T join(Future<T> search) {
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

    private record Timed(String name, List<Document> documents, List<String> queries, long millis) {

        PipelineTrace.Stage stage() {
            return new PipelineTrace.Stage(name, millis, documents.stream().map(PipelineTrace.Hit::of).toList(),
                    queries);
        }
    }

    private static Timed timed(String name, Supplier<List<Document>> search) {
        long start = System.nanoTime();
        List<Document> documents = search.get();
        return new Timed(name, documents, List.of(), millisSince(start));
    }

    private static Timed timedQueries(String name, Supplier<List<String>> transformation) {
        long start = System.nanoTime();
        List<String> queries = transformation.get();
        return new Timed(name, List.of(), queries, millisSince(start));
    }

    private static long millisSince(long start) {
        return TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
    }
}
