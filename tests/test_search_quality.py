"""Tests for search quality improvements.

These tests verify that title and header information improves search relevance.
"""

import pytest

from obsidian_semantic.chunker import Chunk, NoteMetadata, chunk_note, parse_note
from obsidian_semantic.indexer import make_embedding_text


class TestEmbeddingTextGeneration:
    """Test that embedding text includes title and headers for better semantic matching."""

    def test_embedding_text_includes_folder_and_title(self):
        """Folder path and title should be prepended to embedding text."""
        chunk = Chunk(
            id="Programming/Python/Unit Testing.md#chunk_0",
            file_path="Programming/Python/Unit Testing.md",
            title="Unit Testing",
            headers=[],
            text="This note covers testing methodologies.",
            start_line=1,
            end_line=5,
        )

        embedding_text = make_embedding_text(chunk)

        assert "Programming > Python" in embedding_text
        assert "Unit Testing" in embedding_text
        assert "testing methodologies" in embedding_text

    def test_embedding_text_includes_headers(self):
        """Headers breadcrumb should be included in embedding text."""
        chunk = Chunk(
            id="Programming/Python/Unit Testing.md#patching",
            file_path="Programming/Python/Unit Testing.md",
            title="Unit Testing",
            headers=["Patching and Mocking", "MagicMock"],
            text="MagicMock is useful for...",
            start_line=50,
            end_line=60,
        )

        embedding_text = make_embedding_text(chunk)

        # Should contain folder, title and headers in breadcrumb format
        assert "Programming > Python" in embedding_text
        assert "Unit Testing" in embedding_text
        assert "Patching and Mocking" in embedding_text
        assert "MagicMock" in embedding_text
        assert "useful for" in embedding_text

    def test_embedding_text_uses_breadcrumb_format(self):
        """Folder, title, and headers should be joined with ' > ' separator."""
        chunk = Chunk(
            id="Programming/Python/Basics.md#installation",
            file_path="Programming/Python/Basics.md",
            title="Basics",
            headers=["Getting Started", "Installation"],
            text="Install Python using...",
            start_line=10,
            end_line=20,
        )

        embedding_text = make_embedding_text(chunk)

        # Should have breadcrumb format with folder context
        assert "Programming > Python > Basics > Getting Started > Installation" in embedding_text

    def test_embedding_text_without_headers(self):
        """Chunks without headers should have folder and title."""
        chunk = Chunk(
            id="docs/README.md#chunk_0",
            file_path="docs/README.md",
            title="README",
            headers=[],
            text="Welcome to the project.",
            start_line=1,
            end_line=3,
        )

        embedding_text = make_embedding_text(chunk)

        assert "docs > README" in embedding_text
        assert "Welcome to the project" in embedding_text
        # Should NOT have dangling separator
        assert " > \n" not in embedding_text

    def test_embedding_text_root_level_file(self):
        """Files at root level should just have title (no folder)."""
        chunk = Chunk(
            id="README.md#chunk_0",
            file_path="README.md",
            title="README",
            headers=[],
            text="Welcome to the project.",
            start_line=1,
            end_line=3,
        )

        embedding_text = make_embedding_text(chunk)

        assert embedding_text.startswith("README\n")
        assert "Welcome to the project" in embedding_text

    def test_embedding_text_strips_inline_tags(self):
        """Inline tags should be stripped from embedding text."""
        chunk = Chunk(
            id="Programming/Python/Notes.md#chunk_0",
            file_path="Programming/Python/Notes.md",
            title="Notes",
            headers=[],
            text="#python #packaging\n\nSome actual content here.",
            start_line=1,
            end_line=5,
        )

        embedding_text = make_embedding_text(chunk)

        # Tags should be stripped
        assert "#python" not in embedding_text
        assert "#packaging" not in embedding_text
        # Content should remain
        assert "Some actual content here" in embedding_text
        # Folder and title should be present
        assert "Programming > Python > Notes" in embedding_text

    def test_embedding_text_tag_only_chunk(self):
        """Chunks with only tags should still have folder/title context."""
        chunk = Chunk(
            id="Programming/Python/Quick.md#chunk_0",
            file_path="Programming/Python/Quick.md",
            title="Quick",
            headers=[],
            text="#python #quick-tip",
            start_line=1,
            end_line=1,
        )

        embedding_text = make_embedding_text(chunk)

        # Tags stripped, only breadcrumb remains
        assert "#python" not in embedding_text
        assert "Programming > Python > Quick" in embedding_text


class TestShortNoteHandling:
    """Test that short notes are indexed as single chunks."""

    def test_short_note_indexed_as_single_chunk(self):
        """Notes with very little content should be a single chunk."""
        content = """\
# Short Note

#python #quick-tip

Just a brief note about something.
"""
        metadata, body = parse_note(content, "short.md")
        chunks = list(chunk_note(body, "short.md", "Short Note", metadata))

        # Should produce a single chunk, not multiple
        assert len(chunks) == 1
        assert "brief note" in chunks[0].text

    def test_tag_only_intro_not_separate_chunk(self):
        """Intro sections with only tags should not create separate low-quality chunks."""
        content = """\
#python #testing #pytest

## Real Content

This is the actual content with meaningful information about testing.
"""
        metadata, body = parse_note(content, "note.md")
        chunks = list(chunk_note(body, "note.md", "Testing Note", metadata))

        # The first chunk should be meaningful, not just tags
        for chunk in chunks:
            # Check that no chunk is ONLY whitespace/tags
            text_without_tags = chunk.text.replace("#", "").replace("python", "").replace("testing", "").replace("pytest", "")
            assert len(text_without_tags.strip()) > 10 or chunk.headers, f"Chunk too short: {chunk.text!r}"

    def test_normal_sized_note_still_splits(self):
        """Notes with substantial content should still split on headers."""
        content = """\
# Full Note

Introduction with enough content to be meaningful on its own - this paragraph has
plenty of text that makes it a worthwhile chunk to index separately.

## Section One

This section has substantial content about topic one. It includes multiple sentences
and paragraphs worth of information that justify having its own chunk.

## Section Two

Another section with meaningful content about a different topic. This also has enough
text to be a useful standalone chunk for semantic search.
"""
        metadata, body = parse_note(content, "full.md")
        chunks = list(chunk_note(body, "full.md", "Full Note", metadata))

        # Should split into multiple chunks
        assert len(chunks) >= 3
        # Verify headers are captured
        headers_flat = [c.headers for c in chunks]
        assert ["Section One"] in headers_flat
        assert ["Section Two"] in headers_flat


class TestContentLengthCalculation:
    """Test the helper that calculates content length excluding tags."""

    def test_content_length_excludes_inline_tags(self):
        """Tags should not count toward content length."""
        from obsidian_semantic.chunker import _content_length

        text = "#python #testing #pytest"
        assert _content_length(text) == 0

    def test_content_length_counts_real_content(self):
        """Real text should be counted."""
        from obsidian_semantic.chunker import _content_length

        text = "#python Some actual content here"
        length = _content_length(text)
        assert length == len("Some actual content here")

    def test_content_length_handles_mixed_content(self):
        """Mixed tags and content should work correctly."""
        from obsidian_semantic.chunker import _content_length

        text = "#tag1 Real content #tag2 more content #tag3"
        length = _content_length(text)
        # Should only count "Real content" and "more content" parts
        assert length > 20
        assert length < len(text)
