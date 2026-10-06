package com.learnings.rag.eval;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Detects a question that copies wording from its source chunk. Copied phrases make a question unrealistically easy
 * for keyword search, which would bias every comparison the golden set is used for. Identifiers
 * ({@code spring.ai.vectorstore.pgvector.index-type}, {@code ChatClient.Builder}) count as one word: naming the thing
 * you ask about is fine; copying the sentence around it is not.
 */
final class LexicalOverlap {

    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]+(?:[._\\-/:#][\\p{L}\\p{N}]+)*");

    private LexicalOverlap() {
    }

    /** The longest run of consecutive words found in both texts (case-insensitive), or "" when none. */
    static String longestSharedRun(String a, String b) {
        List<String> x = words(a);
        List<String> y = words(b);
        int best = 0;
        int end = 0;
        int[] previous = new int[y.size() + 1];
        for (int i = 1; i <= x.size(); i++) {
            int[] current = new int[y.size() + 1];
            for (int j = 1; j <= y.size(); j++) {
                if (x.get(i - 1).equals(y.get(j - 1))) {
                    current[j] = previous[j - 1] + 1;
                    if (current[j] > best) {
                        best = current[j];
                        end = i;
                    }
                }
            }
            previous = current;
        }
        return String.join(" ", x.subList(end - best, end));
    }

    static List<String> words(String text) {
        List<String> words = new ArrayList<>();
        Matcher matcher = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            words.add(matcher.group());
        }
        return words;
    }

    static int wordCount(String phrase) {
        return words(phrase).size();
    }
}
