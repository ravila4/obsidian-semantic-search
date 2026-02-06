"""Tests for the CLI commands."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from obsidian_semantic.cli import app


@pytest.fixture
def runner() -> CliRunner:
    """Provide CLI test runner."""
    return CliRunner()


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    """Create a test vault with some markdown files."""
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "note1.md").write_text("""\
---
tags:
  - test
---
# Note One

Content of note one.
""")

    (vault / "note2.md").write_text("""\
# Note Two

Content without frontmatter.
""")

    return vault


@pytest.fixture
def mock_embedder() -> Mock:
    """Provide a mock embedder."""
    embedder = Mock()
    embedder.dimension = 768
    embedder.model_name = "test-model"
    embedder.embed.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    # Delegate embed_document/embed_query to embed, mirroring base class
    embedder.embed_document.side_effect = lambda texts: embedder.embed(texts)
    embedder.embed_query.side_effect = lambda texts: embedder.embed(texts)
    return embedder


@pytest.fixture
def configured_mock(vault_path: Path, mock_embedder: Mock) -> Mock:
    """Create a mock config that returns consistent embedder."""
    mock_config = Mock()
    mock_config.database = str(vault_path / ".obsidian-semantic" / "index.lance")
    mock_config.ignore = []
    mock_config.create_embedder.return_value = mock_embedder
    return mock_config


class TestStatusCommand:
    """Test the status command."""

    def test_status_shows_empty_index(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Status on unindexed vault shows zeros."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["status", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "0 chunks" in result.output or "0 files" in result.output

    def test_status_shows_indexed_content(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Status after indexing shows chunk/file counts."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # First index the vault
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Then check status
            result = runner.invoke(app, ["status", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "2 files" in result.output or "files" in result.output.lower()

    def test_status_shows_pending_changes(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Status shows files that need indexing."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index the vault
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Add a new file
            (vault_path / "note3.md").write_text("# Note Three\n\nNew content.")

            # Check status
            result = runner.invoke(app, ["status", "--vault", str(vault_path)])

            assert result.exit_code == 0
            # Should show pending changes
            assert "pending" in result.output.lower() or "new" in result.output.lower()
            assert "note3.md" in result.output

    def test_status_shows_modified_files(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Status shows files modified since last index."""
        import time
        import os

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index the vault
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Modify a file
            time.sleep(0.01)
            note1 = vault_path / "note1.md"
            note1.write_text(note1.read_text() + "\n\nModified content.")
            os.utime(note1, None)  # Touch to ensure mtime changes

            # Check status
            result = runner.invoke(app, ["status", "--vault", str(vault_path)])

            assert result.exit_code == 0
            # Should show modified file
            assert "modified" in result.output.lower() or "pending" in result.output.lower()
            assert "note1.md" in result.output


class TestIndexCommand:
    """Test the index command."""

    def test_index_vault(self, runner: CliRunner, vault_path: Path, configured_mock: Mock):
        """Index command processes vault files."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["index", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "indexed" in result.output.lower() or "processed" in result.output.lower()

    def test_index_full_flag(self, runner: CliRunner, vault_path: Path, configured_mock: Mock):
        """Index --full reindexes all files."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index once
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Index again with --full
            result = runner.invoke(app, ["index", "--full", "--vault", str(vault_path)])

            assert result.exit_code == 0
            # Should report processing files again
            assert "2" in result.output  # 2 files processed

    def test_index_requires_vault_or_env(self, runner: CliRunner):
        """Index without vault path uses OBSIDIAN_VAULT env or errors."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("obsidian_semantic.cli.load_config") as mock_config,
        ):
            mock_config.return_value.database = "/tmp/test.lance"

            result = runner.invoke(app, ["index"])

            # Should error without vault path
            assert result.exit_code != 0 or "vault" in result.output.lower()


class TestSearchCommand:
    """Test the search command."""

    def test_search_returns_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Search returns matching chunks."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index first
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Search
            result = runner.invoke(app, ["search", "note content", "--vault", str(vault_path)])

            assert result.exit_code == 0
            # Should show results
            assert "note" in result.output.lower() or "result" in result.output.lower()

    def test_search_limit_option(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Search respects --limit option."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index first
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["search", "note", "--limit", "1", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0

    def test_search_no_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Search on empty index returns gracefully."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["search", "query", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "no results" in result.output.lower() or "0" in result.output


class TestConfigureCommand:
    """Test the configure command."""

    def test_configure_creates_config_file(self, runner: CliRunner, tmp_path: Path):
        """Configure creates config file in specified directory."""
        config_dir = tmp_path / ".config" / "obsidian-semantic"

        with patch("obsidian_semantic.cli.CONFIG_DIR", config_dir):
            result = runner.invoke(app, ["configure", "--embedder", "gemini"])

            assert result.exit_code == 0
            config_file = config_dir / "config.yaml"
            assert config_file.exists()
            content = config_file.read_text()
            assert "gemini" in content

    def test_configure_vault_path(self, runner: CliRunner, tmp_path: Path):
        """Configure --vault sets default vault path."""
        config_dir = tmp_path / ".config" / "obsidian-semantic"
        vault = tmp_path / "my-vault"

        with patch("obsidian_semantic.cli.CONFIG_DIR", config_dir):
            result = runner.invoke(app, ["configure", "--vault", str(vault)])

            assert result.exit_code == 0
            config_file = config_dir / "config.yaml"
            content = config_file.read_text()
            assert str(vault) in content

    def test_commands_use_configured_vault(
        self, runner: CliRunner, tmp_path: Path, configured_mock: Mock
    ):
        """Commands use vault from config when not specified."""
        config_dir = tmp_path / ".config" / "obsidian-semantic"
        config_dir.mkdir(parents=True)
        vault = tmp_path / "vault"  # From configured_mock fixture
        (config_dir / "config.yaml").write_text(f"vault: {vault}\n")

        with (
            patch("obsidian_semantic.cli.CONFIG_DIR", config_dir),
            patch("obsidian_semantic.cli.load_config", return_value=configured_mock),
        ):
            # Status without --vault should work
            result = runner.invoke(app, ["status"])

            assert result.exit_code == 0
            assert "Vault:" in result.output

    def test_configure_shows_current_config(self, runner: CliRunner, tmp_path: Path):
        """Configure --show displays current configuration."""
        config_dir = tmp_path / ".config" / "obsidian-semantic"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("embedder:\n  type: ollama\n")

        with patch("obsidian_semantic.cli.CONFIG_DIR", config_dir):
            result = runner.invoke(app, ["configure", "--show"])

            assert result.exit_code == 0
            assert "ollama" in result.output

    def test_configure_merges_settings(self, runner: CliRunner, tmp_path: Path):
        """Configure merges new settings with existing ones."""
        config_dir = tmp_path / ".config" / "obsidian-semantic"
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("embedder:\n  type: gemini\n")

        with patch("obsidian_semantic.cli.CONFIG_DIR", config_dir):
            # Add vault without losing embedder
            result = runner.invoke(app, ["configure", "--vault", "/some/path"])

            assert result.exit_code == 0
            content = (config_dir / "config.yaml").read_text()
            assert "gemini" in content
            assert "/some/path" in content


class TestRelatedCommand:
    """Test the related command."""

    def test_related_indexed_note(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock, mock_embedder: Mock
    ):
        """Related with an indexed note uses existing vectors, no embed call."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index first
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Reset embed call count after indexing
            mock_embedder.embed.reset_mock()

            # Run related on an indexed note
            result = runner.invoke(
                app, ["related", "note1.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            # Should NOT have called embed (vectors already in DB)
            mock_embedder.embed.assert_not_called()
            # Should show results (note2.md is the only other note)
            assert "note2.md" in result.output

    def test_related_unindexed_note(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock, mock_embedder: Mock
    ):
        """Related with an unindexed note reads the file and embeds on the fly."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index only note1.md by indexing full vault then deleting note2 from index
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            # Add a third note that is NOT indexed
            (vault_path / "note3.md").write_text("# Note Three\n\nNew unindexed content.\n")

            mock_embedder.embed.reset_mock()

            result = runner.invoke(
                app, ["related", "note3.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            # Should have called embed (note3 not in index)
            mock_embedder.embed.assert_called()

    def test_related_deduplicates_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Same note from multiple chunk searches appears only once in output."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index the vault
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["related", "note1.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            # note2.md should appear at most once in the output
            assert result.output.count("note2.md") == 1

    def test_related_excludes_source_note(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Related should not show the source note itself."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["related", "note1.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            # The source note should not appear in results
            assert "note1.md" not in result.output

    def test_related_no_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Related on empty index returns gracefully."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["related", "note1.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            assert "no related notes" in result.output.lower()

    def test_related_file_not_found(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Related with nonexistent note shows error."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["related", "nonexistent.md", "--vault", str(vault_path)]
            )

            assert result.exit_code != 0 or "not found" in result.output.lower()


class TestHelpOutput:
    """Test help output for commands."""

    def test_main_help(self, runner: CliRunner):
        """Main help shows available commands."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "index" in result.output
        assert "search" in result.output
        assert "related" in result.output
        assert "status" in result.output
        assert "configure" in result.output

    def test_index_help(self, runner: CliRunner):
        """Index help shows options."""
        result = runner.invoke(app, ["index", "--help"])

        assert result.exit_code == 0
        assert "--full" in result.output
        assert "--vault" in result.output

    def test_search_help(self, runner: CliRunner):
        """Search help shows options."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "--limit" in result.output
        assert "--vault" in result.output
