package com.learnings.rag;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * Deterministic bag-of-words embeddings (feature hashing into 1536 buckets). Texts that share words are
 * cosine-similar, which is enough for tests to check that retrieval ranks the obviously relevant chunk first,
 * without calling OpenAI.
 */
public class FakeEmbeddingModel implements EmbeddingModel {

    static final int DIMENSIONS = 1536;

    @Override
    public EmbeddingResponse call(EmbeddingRequest request) {
        List<Embedding> embeddings = new ArrayList<>();
        List<String> texts = request.getInstructions();
        for (int i = 0; i < texts.size(); i++) {
            embeddings.add(new Embedding(vector(texts.get(i)), i));
        }
        return new EmbeddingResponse(embeddings);
    }

    @Override
    public float[] embed(Document document) {
        return vector(document.getText());
    }

    @Override
    public int dimensions() {
        return DIMENSIONS;
    }

    static float[] vector(String text) {
        float[] vector = new float[DIMENSIONS];
        for (String word : text.toLowerCase(Locale.ROOT).split("[^a-z0-9]+")) {
            if (!word.isEmpty()) {
                vector[Math.floorMod(word.hashCode(), DIMENSIONS)] += 1f;
            }
        }
        double norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        if (norm == 0) {
            vector[0] = 1f;
            return vector;
        }
        float length = (float) Math.sqrt(norm);
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= length;
        }
        return vector;
    }
}
