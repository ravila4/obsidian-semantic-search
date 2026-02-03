"""Tests for the markdown chunker."""

import pytest

from obsidian_semantic.chunker import Chunk, NoteMetadata, parse_note


class TestFrontmatterParsing:
    """Test YAML frontmatter extraction."""

    def test_parse_frontmatter_with_tags(self):
        content = """\
---
tags:
  - python
  - testing
created: 2024-01-15
---
# My Note

Some content here.
"""
        metadata, body = parse_note(content, "notes/my-note.md")

        assert metadata.tags == ["python", "testing"]
        assert metadata.created.year == 2024
        assert metadata.created.month == 1
        assert metadata.created.day == 15
        assert "# My Note" in body
        assert "---" not in body

    def test_parse_frontmatter_with_aliases(self):
        content = """\
---
aliases:
  - My Alias
  - Another Name
---
Content here.
"""
        metadata, body = parse_note(content, "note.md")

        assert metadata.aliases == ["My Alias", "Another Name"]

    def test_parse_no_frontmatter(self):
        content = "# Just a heading\n\nSome content."
        metadata, body = parse_note(content, "note.md")

        assert metadata.tags == []
        assert metadata.aliases == []
        assert metadata.created is None
        assert body == content

    def test_parse_empty_frontmatter(self):
        content = """\
---
---
# Heading
"""
        metadata, body = parse_note(content, "note.md")

        assert metadata.tags == []
        assert "# Heading" in body

    def test_inline_tags_extracted(self):
        content = """\
---
tags:
  - frontmatter-tag
---
# Note

Content with #inline-tag and #another-tag here.
"""
        metadata, body = parse_note(content, "note.md")

        assert "frontmatter-tag" in metadata.tags
        assert "inline-tag" in metadata.tags
        assert "another-tag" in metadata.tags

    def test_inline_tags_not_extracted_from_code_blocks(self):
        """Tags inside fenced code blocks should be ignored."""
        content = """\
# Note

Real tag here: #real-tag

```python
# This is a comment, not a tag
color = "#ff0000"  # hex color
```

```css
.class { fill: #f8f9fb; }
```
"""
        metadata, body = parse_note(content, "note.md")

        assert "real-tag" in metadata.tags
        assert "ff0000" not in metadata.tags
        assert "f8f9fb" not in metadata.tags

    def test_inline_tags_not_extracted_from_inline_code(self):
        """Tags inside backtick inline code should be ignored."""
        content = "Use `#define` for macros, but #real-tag is a tag."
        metadata, body = parse_note(content, "note.md")

        assert "real-tag" in metadata.tags
        assert "define" not in metadata.tags


class TestChunking:
    """Test splitting notes into chunks based on headers."""

    def test_split_on_h2_headers(self):
        # Note: content must exceed MIN_NOTE_SIZE_FOR_CHUNKING (200 chars)
        # to trigger section-based chunking
        content = """\
# Main Title

This is an introduction paragraph with enough content to make the note
substantial enough for chunking. We need at least 200 characters of real
content (excluding tags) to trigger the section-splitting behavior.

## Section One

Content of section one with additional text to ensure this section is
meaningful and not just a tiny fragment that would be filtered out.

## Section Two

Content of section two also needs sufficient text to be considered a
valid chunk worth indexing for semantic search purposes.
"""
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "Main Title", metadata))

        assert len(chunks) == 3  # Intro + 2 sections
        assert chunks[0].headers == []
        assert "introduction paragraph" in chunks[0].text
        assert chunks[1].headers == ["Section One"]
        assert "Content of section one" in chunks[1].text
        assert chunks[2].headers == ["Section Two"]

    def test_chunk_ids_are_unique(self):
        content = """\
# Title

## Section A

Text A.

## Section B

Text B.
"""
        metadata, body = parse_note(content, "notes/my-note.md")
        chunks = list(chunk_note(body, "notes/my-note.md", "Title", metadata))

        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"
        assert all("notes/my-note.md#" in id for id in ids)

    def test_fallback_to_h3_for_large_sections(self):
        """If an H2 section exceeds ~2000 chars, split on H3."""
        large_content = "x" * 2500
        content = f"""\
# Title

## Big Section

{large_content}

### Subsection A

Small content A.

### Subsection B

Small content B.
"""
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "Title", metadata))

        # Should split Big Section into subsections
        headers_flat = [c.headers for c in chunks]
        assert ["Big Section", "Subsection A"] in headers_flat
        assert ["Big Section", "Subsection B"] in headers_flat

    def test_line_numbers_tracked(self):
        # Note: content must exceed MIN_NOTE_SIZE_FOR_CHUNKING to trigger splitting
        content = """\
# Title

First paragraph with enough content to make this note substantial.
We need additional text here to ensure the chunking logic kicks in
and splits this note into multiple sections based on the H2 headers.

## Section

Second paragraph with more content to ensure this section is also
meaningful and the line number tracking can be properly verified.
"""
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "Title", metadata))

        # First chunk starts at line 1, second chunk starts at "## Section"
        assert chunks[0].start_line == 1
        assert chunks[1].start_line > chunks[0].end_line

    def test_empty_file_produces_no_chunks(self):
        metadata, body = parse_note("", "empty.md")
        chunks = list(chunk_note(body, "empty.md", "empty", metadata))

        assert chunks == []

    def test_file_with_only_frontmatter(self):
        content = """\
---
tags:
  - empty
---
"""
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "note", metadata))

        assert chunks == []


class TestWikilinkResolution:
    """Test resolving [[wikilinks]] to display text."""

    def test_simple_wikilink_resolved(self):
        content = "See [[Other Note]] for details."
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "note", metadata))

        assert "Other Note" in chunks[0].text
        assert "[[" not in chunks[0].text

    def test_wikilink_with_alias(self):
        content = "Check out [[Long Note Title|short name]] here."
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "note", metadata))

        assert "short name" in chunks[0].text
        assert "Long Note Title" not in chunks[0].text

    def test_wikilink_with_header(self):
        content = "See [[Note#Section]] for more."
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "note", metadata))

        # Should show "Note > Section" or similar
        assert "[[" not in chunks[0].text


# Import chunk_note after parse_note is verified
from obsidian_semantic.chunker import chunk_note
