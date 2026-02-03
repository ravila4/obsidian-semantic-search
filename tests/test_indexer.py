"""Tests for the vault indexer."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from obsidian_semantic.indexer import VaultIndexer, IndexResult


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    """Create a test vault with some markdown files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    # Create some test notes
    (vault / "note1.md").write_text("""\
---
tags:
  - test
---
# Note One

## Section A

Content of section A.

## Section B

Content of section B.
""")

    (vault / "note2.md").write_text("""\
# Note Two

Some content without frontmatter.
""")

    # Create a subdirectory with a note
    (vault / "projects").mkdir()
    (vault / "projects" / "project1.md").write_text("""\
---
tags:
  - project
---
# Project One

## Overview

Project overview here.
""")

    # Create files that should be ignored
    (vault / ".obsidian").mkdir()
    (vault / ".obsidian" / "config.json").write_text("{}")
    (vault / ".git").mkdir()
    (vault / ".git" / "config").write_text("")

    return vault


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test_db"


@pytest.fixture
def mock_embedder() -> Mock:
    """Provide a mock embedder."""
    embedder = Mock()
    embedder.dimension = 768
    embedder.model_name = "test-model"
    embedder.embed.return_value = [[0.1] * 768]  # Single embedding
    return embedder


class TestVaultTraversal:
    """Test vault file discovery."""

    def test_finds_markdown_files(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should find all .md files in vault."""
        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        files = list(indexer._discover_files())

        paths = [f.relative_to(vault_path) for f in files]
        assert Path("note1.md") in paths
        assert Path("note2.md") in paths
        assert Path("projects/project1.md") in paths

    def test_ignores_obsidian_directory(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should skip .obsidian directory."""
        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        files = list(indexer._discover_files())

        for f in files:
            assert ".obsidian" not in str(f)

    def test_ignores_git_directory(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should skip .git directory."""
        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        files = list(indexer._discover_files())

        for f in files:
            assert ".git" not in str(f)

    def test_respects_custom_ignore_patterns(
        self, vault_path: Path, db_path: Path, mock_embedder: Mock
    ):
        """Should respect custom ignore patterns."""
        # Create a file that matches custom pattern
        (vault_path / "draft-note.md").write_text("# Draft")

        indexer = VaultIndexer(
            vault_path, db_path, mock_embedder, ignore_patterns=["draft-*"]
        )
        files = list(indexer._discover_files())

        for f in files:
            assert "draft-" not in f.name


class TestChangeDetection:
    """Test incremental indexing via mtime comparison."""

    def test_detects_new_files(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should identify files not yet in database."""
        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # First run - all files are new
        files = list(indexer._discover_files())
        changes = indexer._detect_changes(files)

        assert len(changes["new"]) == 3
        assert len(changes["modified"]) == 0
        assert len(changes["deleted"]) == 0

    def test_detects_modified_files(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should identify files modified since last index."""
        # Set up mock to return correct number of embeddings
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # Run initial index
        indexer.index()

        # Modify a file - use os.utime to ensure mtime changes
        import os
        import time
        time.sleep(0.01)
        note1 = vault_path / "note1.md"
        note1.write_text(note1.read_text() + "\n\nNew content added.")
        # Touch the file to ensure mtime is updated
        os.utime(note1, None)

        # Check for changes
        files = list(indexer._discover_files())
        changes = indexer._detect_changes(files)

        assert len(changes["modified"]) == 1
        assert changes["modified"][0].name == "note1.md"

    def test_detects_deleted_files(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should identify files deleted from vault."""
        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # Run initial index
        indexer.index()

        # Delete a file
        (vault_path / "note2.md").unlink()

        # Check for changes
        files = list(indexer._discover_files())
        changes = indexer._detect_changes(files)

        assert len(changes["deleted"]) == 1
        assert "note2.md" in changes["deleted"][0]


class TestIndexing:
    """Test the full indexing workflow."""

    def test_full_index(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should index all files on first run."""
        # Make embedder return correct number of embeddings
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        result = indexer.index()

        assert result.files_processed == 3
        assert result.chunks_created > 0
        assert result.errors == []

    def test_incremental_index(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should only process changed files on subsequent runs."""
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # First run
        result1 = indexer.index()
        assert result1.files_processed == 3

        # Second run with no changes
        result2 = indexer.index()
        assert result2.files_processed == 0

    def test_full_reindex_flag(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Should reindex all files when full=True."""
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # First run
        indexer.index()

        # Full reindex
        result = indexer.index(full=True)
        assert result.files_processed == 3

    def test_index_removes_deleted_file_chunks(
        self, vault_path: Path, db_path: Path, mock_embedder: Mock
    ):
        """Should remove chunks when file is deleted."""
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)

        # First run
        indexer.index()
        stats1 = indexer._db.get_stats()

        # Delete a file
        (vault_path / "note2.md").unlink()

        # Incremental index
        result = indexer.index()
        stats2 = indexer._db.get_stats()

        assert stats2.file_count < stats1.file_count
        assert result.files_deleted == 1

    def test_index_handles_parse_errors(
        self, vault_path: Path, db_path: Path, mock_embedder: Mock
    ):
        """Should continue indexing after parse errors."""
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        # Create a file with invalid frontmatter
        (vault_path / "bad.md").write_text("""\
---
tags: [unclosed
---
# Bad Note
""")

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        result = indexer.index()

        # Should still process other files
        assert result.files_processed >= 3
        # Errors are collected but don't stop indexing
        # (bad frontmatter is handled gracefully by the chunker)


class TestIndexResult:
    """Test index result data."""

    def test_result_contains_timing(self, vault_path: Path, db_path: Path, mock_embedder: Mock):
        """Index result should include duration."""
        mock_embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        indexer = VaultIndexer(vault_path, db_path, mock_embedder)
        result = indexer.index()

        assert result.duration_seconds >= 0
        assert result.started_at is not None
        assert result.finished_at is not None
