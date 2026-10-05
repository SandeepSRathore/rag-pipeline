#!/usr/bin/env bash
# Downloads the curated Spring AI reference pages listed in scripts/corpus-pages.txt into corpus/.
# Usage: scripts/fetch-corpus.sh            (tag v2.0.1)
#        SPRING_AI_DOCS_TAG=v2.0.2 scripts/fetch-corpus.sh
set -euo pipefail

TAG="${SPRING_AI_DOCS_TAG:-v2.0.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/corpus"
PAGES="spring-ai-docs/src/main/antora/modules/ROOT/pages"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Fetching Spring AI docs at $TAG (sparse, docs only)…"
git clone --quiet --depth 1 --branch "$TAG" --filter=blob:none --sparse \
    https://github.com/spring-projects/spring-ai.git "$WORK/spring-ai"
git -C "$WORK/spring-ai" sparse-checkout set "$PAGES"

rm -rf "$DEST"
mkdir -p "$DEST"
copied=0
missing=0
while IFS= read -r page || [[ -n "$page" ]]; do
    [[ -z "$page" || "$page" == \#* ]] && continue
    src="$WORK/spring-ai/$PAGES/$page"
    if [[ -f "$src" ]]; then
        mkdir -p "$DEST/$(dirname "$page")"
        cp "$src" "$DEST/$page"
        copied=$((copied + 1))
    else
        echo "WARN: $page not found at $TAG" >&2
        missing=$((missing + 1))
    fi
done < "$ROOT/scripts/corpus-pages.txt"

echo "Copied $copied pages into corpus/ ($missing missing)."
[[ "$copied" -gt 0 ]]
