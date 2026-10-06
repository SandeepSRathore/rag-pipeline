package com.learnings.rag.eval;

import java.nio.file.Path;
import java.util.List;

/** A validated golden set. @param sha256 of the file, recorded in every report so results stay traceable */
public record GoldenSet(Path path, String sha256, List<GoldenItem> items) {
}
