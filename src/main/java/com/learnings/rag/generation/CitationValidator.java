package com.learnings.rag.generation;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Checks an answer's {@code [n]} citations against the number of sources it was given. Code is skipped (fenced
 * blocks, including an unterminated one, and inline code), where {@code parts[1]} is an index, not a citation; the
 * UI's chips don't make that distinction.
 */
public final class CitationValidator {

    private static final Pattern FENCED_CODE = Pattern.compile("(?s)```.*?(?:```|$)");
    private static final Pattern INLINE_CODE = Pattern.compile("`[^`\\n]*`");
    private static final Pattern CITATION = Pattern.compile("\\[(\\d{1,4})]");

    private CitationValidator() {
    }

    /**
     * @param cited the distinct source numbers cited, in order of first appearance
     * @param outOfRange the cited numbers with no matching source
     */
    public record Check(List<Integer> cited, List<Integer> outOfRange) {

        /** At least one citation, and every one points at a source. */
        public boolean valid() {
            return !cited.isEmpty() && outOfRange.isEmpty();
        }
    }

    public static Check check(String answer, int sourceCount) {
        String prose = INLINE_CODE.matcher(FENCED_CODE.matcher(answer).replaceAll(" ")).replaceAll(" ");
        Set<Integer> cited = new LinkedHashSet<>();
        Matcher citation = CITATION.matcher(prose);
        while (citation.find()) {
            cited.add(Integer.parseInt(citation.group(1)));
        }
        List<Integer> outOfRange = cited.stream().filter(n -> n < 1 || n > sourceCount).toList();
        return new Check(List.copyOf(cited), outOfRange);
    }
}
