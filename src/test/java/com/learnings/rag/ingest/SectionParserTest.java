package com.learnings.rag.ingest;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.learnings.rag.ingest.SectionParser.ParsedText;
import com.learnings.rag.ingest.SectionParser.Section;

class SectionParserTest {

    @Test
    void titleComesFromTheFirstLevelOneHeadingAndThePreambleHasAnEmptyPath() {
        ParsedText parsed = SectionParser.parse("""
                = PGvector

                PGvector is a PostgreSQL extension.
                """);

        assertThat(parsed.title()).isEqualTo("PGvector");
        assertThat(parsed.sections()).containsExactly(new Section(List.of(), "PGvector is a PostgreSQL extension."));
    }

    @Test
    void nestedAsciidocHeadingsBuildBreadcrumbPaths() {
        ParsedText parsed = SectionParser.parse("""
                = Doc

                == A

                alpha

                === A1

                beta

                == B

                gamma
                """);

        assertThat(parsed.sections()).containsExactly(
                new Section(List.of("A"), "alpha"),
                new Section(List.of("A", "A1"), "beta"),
                new Section(List.of("B"), "gamma"));
    }

    @Test
    void markdownHeadingsWorkTheSameWay() {
        ParsedText parsed = SectionParser.parse("""
                # Guide

                ## Install

                run it

                ### Maven

                mvn install
                """);

        assertThat(parsed.title()).isEqualTo("Guide");
        assertThat(parsed.sections()).containsExactly(
                new Section(List.of("Install"), "run it"),
                new Section(List.of("Install", "Maven"), "mvn install"));
    }

    @Test
    void headingLikeLinesInsideListingsStayInTheBody() {
        ParsedText parsed = SectionParser.parse("""
                == Usage

                [source,bash]
                ----
                == not a heading
                # not one either
                ----
                """);

        assertThat(parsed.sections()).singleElement().satisfies(section -> {
            assertThat(section.path()).containsExactly("Usage");
            assertThat(section.body()).contains("== not a heading", "# not one either");
        });
    }

    @Test
    void textWithoutALevelOneHeadingHasNoTitle() {
        assertThat(SectionParser.parse("== Only\n\ntext").title()).isNull();
    }

    @Test
    void headingOnlySectionsContributeToThePathButAreNotEmitted() {
        assertThat(SectionParser.parse("== Parent\n\n=== Child\n\ntext").sections())
                .containsExactly(new Section(List.of("Parent", "Child"), "text"));
    }

    @Test
    void asciidocAttributeAndAnchorLinesAreDropped() {
        ParsedText parsed = SectionParser.parse("""
                [[pgvector]]
                = PGvector
                :page-toc: true
                [#intro]
                Hello.
                """);

        assertThat(parsed.sections()).containsExactly(new Section(List.of(), "Hello."));
    }

    @Test
    void laterLevelOneHeadingsActAsTopLevelSections() {
        ParsedText parsed = SectionParser.parse("# One\n\na\n\n# Two\n\nb");

        assertThat(parsed.title()).isEqualTo("One");
        assertThat(parsed.sections()).containsExactly(
                new Section(List.of(), "a"),
                new Section(List.of("Two"), "b"));
    }

    @Test
    void fenceKeysRecognizeListingsLiteralsTablesAndMarkdownFences() {
        assertThat(SectionParser.fenceKey("----")).isEqualTo("----");
        assertThat(SectionParser.fenceKey("......")).isEqualTo("......");
        assertThat(SectionParser.fenceKey("|===")).isEqualTo("|===");
        assertThat(SectionParser.fenceKey("```java")).isEqualTo("```");
        assertThat(SectionParser.fenceKey("-- not a fence")).isNull();
    }
}
