package com.learnings.rag.ingest;

import java.util.ArrayList;
import java.util.List;

import org.springframework.ai.document.Document;
import org.springframework.ai.tokenizer.JTokkitTokenCountEstimator;
import org.springframework.ai.tokenizer.TokenCountEstimator;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.learnings.rag.config.RagProperties;

/**
 * Turns a document into retrieval-sized chunks that respect its structure:
 * <ol>
 * <li>split into sections at headings ({@link SectionParser});</li>
 * <li>split each section into blocks at blank lines, keeping every code listing and table whole;</li>
 * <li>pack blocks into chunks of at most {@code maxTokens}, repeating trailing prose up to {@code overlapTokens};</li>
 * <li>merge chunks under {@code minTokens} into the next one, under their common heading.</li>
 * </ol>
 */
@Component
public class StructureAwareChunker {

    /** Bump when the algorithm changes, so every document is re-chunked on the next ingest. */
    static final int ALGORITHM_VERSION = 1;

    private final RagProperties.Chunking settings;
    private final TokenCountEstimator tokens = new JTokkitTokenCountEstimator();
    private final TokenTextSplitter oversizedProseSplitter;

    @Autowired
    public StructureAwareChunker(RagProperties properties) {
        this(properties.chunking());
    }

    public StructureAwareChunker(RagProperties.Chunking settings) {
        this.settings = settings;
        this.oversizedProseSplitter = TokenTextSplitter.builder()
                .withChunkSize(settings.maxTokens())
                .withMinChunkLengthToEmbed(1) // never drop a short tail: that would silently lose text
                .withKeepSeparator(true)
                .build();
    }

    /** Identifies the chunking configuration; part of each document's fingerprint. */
    public String fingerprint() {
        return "v" + ALGORITHM_VERSION + ":" + settings;
    }

    public ChunkedText chunk(String text, String fallbackTitle) {
        SectionParser.ParsedText parsed = SectionParser.parse(text);
        String title = parsed.title() != null ? parsed.title() : fallbackTitle;

        List<Piece> pieces = new ArrayList<>();
        for (SectionParser.Section section : parsed.sections()) {
            for (String body : pack(blocks(section.body()))) {
                pieces.add(new Piece(section.path(), body));
            }
        }

        List<Piece> merged = mergeSmall(pieces);
        List<Chunk> chunks = new ArrayList<>(merged.size());
        for (int i = 0; i < merged.size(); i++) {
            Piece piece = merged.get(i);
            chunks.add(new Chunk(i, piece.path(), piece.body(), tokens.estimate(piece.body())));
        }
        return new ChunkedText(title, List.copyOf(chunks));
    }

    private record Piece(List<String> path, String body) {
    }

    /** @param atomic a code listing or table, which is never split or used as overlap */
    private record Block(String text, boolean atomic, int tokens) {
    }

    private List<Block> blocks(String body) {
        List<Block> blocks = new ArrayList<>();
        List<String> lines = new ArrayList<>();
        boolean atomic = false;
        String openFence = null;
        for (String line : body.split("\n", -1)) {
            String fence = SectionParser.fenceKey(line);
            if (openFence == null && line.isBlank()) {
                addBlock(blocks, lines, atomic);
                lines.clear();
                atomic = false;
                continue;
            }
            lines.add(line);
            if (openFence == null && fence != null) {
                openFence = fence;
                atomic = true;
            }
            else if (openFence != null && openFence.equals(fence)) {
                openFence = null;
            }
        }
        addBlock(blocks, lines, atomic);
        return blocks;
    }

    private void addBlock(List<Block> blocks, List<String> lines, boolean atomic) {
        String text = String.join("\n", lines).stripTrailing();
        if (!text.isEmpty()) {
            blocks.add(new Block(text, atomic, tokens.estimate(text)));
        }
    }

    private List<String> pack(List<Block> blocks) {
        List<String> out = new ArrayList<>();
        List<Block> window = new ArrayList<>();
        int windowTokens = 0;
        for (Block block : blocks) {
            if (block.tokens() > settings.maxTokens()) {
                emit(out, window);
                window = new ArrayList<>();
                windowTokens = 0;
                if (block.atomic()) {
                    out.add(block.text());
                }
                else {
                    oversizedProseSplitter.split(new Document(block.text())).forEach(d -> out.add(d.getText()));
                }
                continue;
            }
            if (!window.isEmpty() && windowTokens + block.tokens() > settings.maxTokens()) {
                emit(out, window);
                window = overlapTail(window);
                windowTokens = window.stream().mapToInt(Block::tokens).sum();
                if (windowTokens + block.tokens() > settings.maxTokens()) {
                    window = new ArrayList<>();
                    windowTokens = 0;
                }
            }
            window.add(block);
            windowTokens += block.tokens();
        }
        emit(out, window);
        return out;
    }

    private List<Block> overlapTail(List<Block> window) {
        List<Block> tail = new ArrayList<>();
        int total = 0;
        for (int i = window.size() - 1; i >= 0; i--) {
            Block block = window.get(i);
            if (block.atomic() || total + block.tokens() > settings.overlapTokens()) {
                break;
            }
            tail.addFirst(block);
            total += block.tokens();
        }
        return tail;
    }

    private static void emit(List<String> out, List<Block> window) {
        if (!window.isEmpty()) {
            out.add(String.join("\n\n", window.stream().map(Block::text).toList()));
        }
    }

    private List<Piece> mergeSmall(List<Piece> pieces) {
        List<Piece> out = new ArrayList<>();
        Piece pending = null;
        for (Piece piece : pieces) {
            Piece current = piece;
            if (pending != null) {
                Piece merged = new Piece(commonPrefix(pending.path(), piece.path()), pending.body() + "\n\n" + piece.body());
                if (tokens.estimate(merged.body()) <= settings.maxTokens()) {
                    current = merged;
                }
                else {
                    out.add(pending);
                }
                pending = null;
            }
            if (tokens.estimate(current.body()) < settings.minTokens()) {
                pending = current;
            }
            else {
                out.add(current);
            }
        }
        if (pending != null) {
            out.add(pending);
        }
        return out;
    }

    private static List<String> commonPrefix(List<String> a, List<String> b) {
        int n = 0;
        while (n < a.size() && n < b.size() && a.get(n).equals(b.get(n))) {
            n++;
        }
        return List.copyOf(a.subList(0, n));
    }
}
