"""Tests for the CLI commands."""

import json
import re
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
    from obsidian_semantic.config import SuggestLinksConfig

    mock_config = Mock()
    mock_config.database = str(vault_path / ".obsidian-semantic" / "index.lance")
    mock_config.ignore = []
    mock_config.create_embedder.return_value = mock_embedder
    mock_config.suggest_links = SuggestLinksConfig()
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

    def test_status_always_prints_pending_counts(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Status prints a Pending counts line even when nothing is pending."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            index_result = runner.invoke(app, ["index", "--vault", str(vault_path)])
            assert index_result.exit_code == 0, index_result.output

            result = runner.invoke(app, ["status", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "Pending:" in result.output
            assert "0 new" in result.output
            assert "0 modified" in result.output
            assert "0 deleted" in result.output

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

    def test_search_json_no_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """--json with empty index returns [] not plain text."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["search", "query", "--json", "--vault", str(vault_path)])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data == []

    def test_search_json_output(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """--json flag produces valid JSON with expected fields."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["search", "note content", "--json", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) > 0
            first = data[0]
            assert "file_path" in first
            assert "score" in first
            assert "text" in first
            assert "headers" in first
            assert "chunk_id" in first

    def test_search_warns_when_folder_is_ignored(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Searching with --folder X where X matches an ignore pattern surfaces a stderr hint."""
        configured_mock.ignore = ["Templates/*"]
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                ["search", "anything", "--folder", "Templates", "--vault", str(vault_path)],
            )
            assert result.exit_code == 0
            assert "excluded from indexing" in result.stderr.lower()
            assert "Templates" in result.stderr
            # Pointing the user at the config is the whole point of the warning.
            assert "config" in result.stderr.lower()

    def test_search_no_warning_for_indexed_folder(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """A normal --folder filter doesn't trigger the ignore warning."""
        configured_mock.ignore = ["Templates/*"]
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                ["search", "anything", "--folder", "Programming", "--vault", str(vault_path)],
            )
            assert result.exit_code == 0
            assert "excluded from indexing" not in result.stderr.lower()

    def test_search_warns_for_default_dotfolders(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """Built-in default ignores (.obsidian, .git) also trigger the warning."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                ["search", "anything", "--folder", ".obsidian", "--vault", str(vault_path)],
            )
            assert result.exit_code == 0
            assert "excluded from indexing" in result.stderr.lower()


@pytest.fixture
def multi_chunk_vault(tmp_path: Path) -> Path:
    """Vault with one note that produces multiple chunks (split on H2).

    Verifies the chunker actually emits multiple chunks for big_note.md so
    downstream per-file dedup tests aren't trivially passing on a single chunk.
    """
    from obsidian_semantic.chunker import NoteMetadata, chunk_note

    vault = tmp_path / "multi_vault"
    vault.mkdir()
    body = (
        "## Section A\n"
        + ("Detailed content about pythons and snakes. " * 8)
        + "\n\n"
        + "## Section B\n"
        + ("More detailed material covering pythons. " * 8)
        + "\n\n"
        + "## Section C\n"
        + ("Yet another long section discussing pythons. " * 8)
        + "\n"
    )
    (vault / "big_note.md").write_text(body)
    (vault / "other.md").write_text("# Other\n\nSomething different here.\n")

    chunks = list(chunk_note(body, "big_note.md", "big_note", NoteMetadata()))
    assert len(chunks) >= 2, (
        f"fixture is broken: big_note.md should produce multiple chunks, got {len(chunks)}"
    )
    return vault


def _mk_result(file_path: str, score: float, idx: int = 0):
    """Construct a SearchResult for unit-testing post-filters."""
    from obsidian_semantic.db import SearchResult

    return SearchResult(
        chunk_id=f"{file_path}#chunk_{idx}",
        file_path=file_path,
        title=Path(file_path).stem,
        headers=[],
        text=f"chunk {idx} of {file_path}",
        score=score,
        start_line=idx * 10,
    )


class TestLimitPerFile:
    """Unit tests for the pure _limit_per_file helper.

    Tests the post-filter directly with hand-crafted SearchResults so the
    assertions don't depend on the mock embedder producing varied scores.
    """

    def test_caps_at_one_per_file(self):
        from obsidian_semantic.cli import _limit_per_file

        results = [
            _mk_result("a.md", 0.9, 0),
            _mk_result("a.md", 0.8, 1),
            _mk_result("b.md", 0.7, 0),
            _mk_result("a.md", 0.6, 2),
        ]
        out = _limit_per_file(results, 1)

        assert [r.file_path for r in out] == ["a.md", "b.md"]

    def test_caps_at_n_per_file(self):
        from obsidian_semantic.cli import _limit_per_file

        results = [
            _mk_result("a.md", 0.9, 0),
            _mk_result("a.md", 0.8, 1),
            _mk_result("a.md", 0.7, 2),  # 3rd a.md → dropped at per_file=2
            _mk_result("b.md", 0.6, 0),
        ]
        out = _limit_per_file(results, 2)

        paths = [r.file_path for r in out]
        assert paths == ["a.md", "a.md", "b.md"]

    def test_preserves_input_order(self):
        """First-seen wins per file (relevant when LanceDB returns sorted)."""
        from obsidian_semantic.cli import _limit_per_file

        results = [
            _mk_result("a.md", 0.9, 0),  # highest-scoring a.md
            _mk_result("b.md", 0.8, 0),
            _mk_result("a.md", 0.5, 1),  # lower a.md, dropped at per_file=1
        ]
        out = _limit_per_file(results, 1)

        assert [r.score for r in out] == [0.9, 0.8]

    def test_empty_input(self):
        from obsidian_semantic.cli import _limit_per_file

        assert _limit_per_file([], 1) == []


class TestSearchPerFileAndScoreMin:
    """Test --per-file dedup and --score-min threshold."""

    def _configure(self, mock: Mock, vault: Path) -> Mock:
        mock.database = str(vault / ".obsidian-semantic" / "index.lance")
        return mock

    def test_default_dedups_per_file(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """Default search returns each file at most once."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            result = runner.invoke(
                app,
                ["search", "python", "--json", "--vault", str(multi_chunk_vault)],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            paths = [r["file_path"] for r in data]
            assert len(paths) == len(set(paths)), (
                f"expected unique file paths by default, got {paths}"
            )

    def test_per_file_zero_returns_all_chunks(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """--per-file 0 disables dedup; multi-chunk note appears multiple times."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            result = runner.invoke(
                app,
                [
                    "search",
                    "python",
                    "--per-file",
                    "0",
                    "--json",
                    "--vault",
                    str(multi_chunk_vault),
                ],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            big_count = sum(1 for r in data if r["file_path"] == "big_note.md")
            assert big_count > 1, (
                f"big_note.md should appear multiple times with --per-file 0, "
                f"got {big_count} (paths: {[r['file_path'] for r in data]})"
            )

    def test_per_file_two_caps_at_two(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """--per-file 2 returns at most 2 chunks per file."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            result = runner.invoke(
                app,
                [
                    "search",
                    "python",
                    "--per-file",
                    "2",
                    "--json",
                    "--vault",
                    str(multi_chunk_vault),
                ],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            big_count = sum(1 for r in data if r["file_path"] == "big_note.md")
            assert big_count <= 2, f"expected <=2 chunks for big_note.md, got {big_count}"

    def test_score_min_filters_below_threshold(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """--score-min above the achievable score returns no results."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            # Mock embedder yields identical vectors → score == 1.0;
            # threshold > 1.0 must drop everything.
            result = runner.invoke(
                app,
                [
                    "search",
                    "python",
                    "--score-min",
                    "1.5",
                    "--json",
                    "--vault",
                    str(multi_chunk_vault),
                ],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data == []

    def test_score_min_passes_through(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """--score-min below achievable score returns results."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            result = runner.invoke(
                app,
                [
                    "search",
                    "python",
                    "--score-min",
                    "0.5",
                    "--json",
                    "--vault",
                    str(multi_chunk_vault),
                ],
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) > 0
            assert all(r["score"] >= 0.5 for r in data)

    def test_search_help_includes_new_flags(self, runner: CliRunner):
        """Help text mentions --per-file and --score-min."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "--per-file" in result.output
        assert "--score-min" in result.output

    def test_limit_zero_is_rejected(
        self, runner: CliRunner, multi_chunk_vault: Path, configured_mock: Mock
    ):
        """--limit 0 errors out instead of silently returning nothing."""
        self._configure(configured_mock, multi_chunk_vault)
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(multi_chunk_vault)])
            result = runner.invoke(
                app,
                [
                    "search",
                    "python",
                    "--limit",
                    "0",
                    "--vault",
                    str(multi_chunk_vault),
                ],
            )

            assert result.exit_code != 0
            assert "limit" in result.output.lower()


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

    def test_related_json_no_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """--json with empty index returns [] not plain text."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["related", "note1.md", "--json", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data == []

    def test_related_json_output(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """--json flag on related produces valid JSON with expected fields."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["related", "note1.md", "--json", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) > 0
            first = data[0]
            assert "file_path" in first
            assert "score" in first
            assert "title" in first
            assert "text" in first


class TestShowCommand:
    """Test the show command."""

    def test_show_direct_relative_path(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show prints contents when given a relative path."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["show", "note1.md", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "Content of note one." in result.output
            assert "Note One" in result.output

    def test_show_resolves_basename_in_subfolder(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show finds a note by basename when it lives in a subfolder."""
        sub = vault_path / "Sub Folder"
        sub.mkdir()
        (sub / "Fishers Test.md").write_text("# Fishers\n\nSubfolder body.\n")

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "Fishers Test.md", "--vault", str(vault_path)]
            )

            assert result.exit_code == 0
            assert "Subfolder body." in result.output

    def test_show_adds_md_extension(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show appends .md if the user omits it."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["show", "note1", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "Content of note one." in result.output

    def test_show_file_not_found(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show errors out to stderr when the note can't be found."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "does-not-exist.md", "--vault", str(vault_path)]
            )

            assert result.exit_code != 0
            assert "not found" in result.stderr.lower()

    def test_show_ambiguous_match(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show errors out to stderr and lists candidates when basename matches multiple files."""
        sub_a = vault_path / "A"
        sub_b = vault_path / "B"
        sub_a.mkdir()
        sub_b.mkdir()
        (sub_a / "duplicate.md").write_text("# A\n")
        (sub_b / "duplicate.md").write_text("# B\n")

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "duplicate.md", "--vault", str(vault_path)]
            )

            assert result.exit_code != 0
            assert "A/duplicate.md" in result.stderr
            assert "B/duplicate.md" in result.stderr

    def test_show_skips_dot_directories(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """show ignores files inside .obsidian/ and other dot-dirs when resolving names."""
        obsidian = vault_path / ".obsidian" / "plugins"
        obsidian.mkdir(parents=True)
        (obsidian / "note1.md").write_text("# Plugin garbage\n")

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(app, ["show", "note1.md", "--vault", str(vault_path)])

            assert result.exit_code == 0
            assert "Content of note one." in result.output
            assert "Plugin garbage" not in result.output

    @pytest.fixture
    def anchor_vault(self, tmp_path: Path) -> Path:
        """Vault with notes designed to exercise heading-anchor matching."""
        vault = tmp_path / "anchor-vault"
        vault.mkdir()
        (vault / "nested.md").write_text(
            "---\n"
            "tags: [test]\n"
            "---\n"
            "# Title\n"
            "\n"
            "## Setup\n"
            "Setup body line.\n"
            "\n"
            "### Installation\n"
            "Install steps.\n"
            "\n"
            "#### Linux\n"
            "Linux details.\n"
            "\n"
            "### Configuration\n"
            "Config steps.\n"
            "\n"
            "## Other\n"
            "Other body.\n"
        )
        (vault / "code.md").write_text(
            "## Real Heading\n"
            "Real body.\n"
            "\n"
            "```python\n"
            "## fake heading inside fence\n"
            "x = 1\n"
            "```\n"
            "\n"
            "## Next Real\n"
            "Next body.\n"
        )
        (vault / "noheadings.md").write_text(
            "Just a paragraph with no headings at all.\n"
        )
        (vault / "dupes.md").write_text(
            "## Alpha\n"
            "### Repeat\n"
            "first body.\n"
            "\n"
            "## Beta\n"
            "### Repeat\n"
            "second body.\n"
        )
        return vault

    def test_show_prints_section_by_anchor(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """`show note#Heading` prints just that section."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "nested#Other", "--vault", str(anchor_vault)]
            )
            assert result.exit_code == 0
            assert "## Other" in result.output
            assert "Other body." in result.output
            assert "Setup body line." not in result.output

    def test_show_section_includes_heading_line_and_stops_at_next_sibling(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Section output starts at the heading line and ends before the next sibling/parent."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "nested#Setup", "--vault", str(anchor_vault)]
            )
            assert result.exit_code == 0
            assert "## Setup" in result.output
            assert "Setup body line." in result.output
            # Nested headings inside Setup should be included
            assert "### Installation" in result.output
            assert "### Configuration" in result.output
            # The next H2 sibling must NOT bleed in
            assert "## Other" not in result.output
            assert "Other body." not in result.output

    def test_show_section_case_insensitive(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Heading match is case-insensitive."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "nested#sEtUp", "--vault", str(anchor_vault)]
            )
            assert result.exit_code == 0
            assert "Setup body line." in result.output

    def test_show_section_nested(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """`note#Parent#Child` returns only the child sub-section."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                ["show", "nested#Setup#Installation", "--vault", str(anchor_vault)],
            )
            assert result.exit_code == 0
            assert "### Installation" in result.output
            assert "Install steps." in result.output
            # Sibling H3 must not appear
            assert "### Configuration" not in result.output
            assert "Config steps." not in result.output
            # Parent body must not appear
            assert "Setup body line." not in result.output

    def test_show_section_suffix_matches_deep_breadcrumb(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """`note#Installation` finds a section whose breadcrumb is `Setup > Installation`."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "nested#Installation", "--vault", str(anchor_vault)]
            )
            assert result.exit_code == 0
            assert "### Installation" in result.output
            assert "Install steps." in result.output

    def test_show_section_h4_reachable(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """H4 headings are reachable via anchors (chunker-independent)."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                [
                    "show",
                    "nested#Setup#Installation#Linux",
                    "--vault",
                    str(anchor_vault),
                ],
            )
            assert result.exit_code == 0
            assert "#### Linux" in result.output
            assert "Linux details." in result.output
            # Sibling H3 below the H4 must not appear
            assert "### Configuration" not in result.output

    def test_show_section_inside_code_block_ignored(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """A `## ...` line inside a fenced code block is not treated as a section."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app,
                ["show", "code#fake heading inside fence", "--vault", str(anchor_vault)],
            )
            assert result.exit_code != 0
            assert "Section not found" in result.stderr

            # The two real headings DO match.
            real = runner.invoke(
                app, ["show", "code#Real Heading", "--vault", str(anchor_vault)]
            )
            assert real.exit_code == 0
            assert "Real body." in real.output
            # The fake heading line lives inside a code block but should not act as
            # a section boundary, so it stays embedded in the section output.
            assert "## fake heading inside fence" in real.output
            # And we must stop before the next REAL H2.
            assert "Next body." not in real.output

    def test_show_section_no_headings_in_note(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Anchoring into a note with no headings produces a specific error."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "noheadings#anything", "--vault", str(anchor_vault)]
            )
            assert result.exit_code != 0
            assert "no addressable headings" in result.stderr.lower()

    def test_show_section_not_found_lists_available(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """A bad anchor lists available section breadcrumbs on stderr."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "nested#does-not-exist", "--vault", str(anchor_vault)]
            )
            assert result.exit_code != 0
            assert "Section not found" in result.stderr
            assert "does-not-exist" in result.stderr
            assert "Available sections:" in result.stderr
            # Listings are printed in canonical 'note#A#B' form so they can be
            # copy-pasted directly back into a `show` invocation.
            assert "nested#Title#Setup#Installation" in result.stderr
            assert "nested#Title#Other" in result.stderr

    def test_show_section_listing_roundtrips_through_parser(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """A path printed in the 'Available sections' listing must parse cleanly when fed back in.

        Roundtrip contract: the error output is the parser's input format. If this
        ever regresses (e.g. someone reintroduces ' > ' as the separator) this test
        catches it before users do.
        """
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            bad = runner.invoke(
                app, ["show", "nested#does-not-exist", "--vault", str(anchor_vault)]
            )
            assert bad.exit_code != 0

            listed = [
                line.strip()
                for line in bad.stderr.splitlines()
                if line.startswith("  ") and "#" in line
            ]
            assert listed, "Expected at least one section listing on stderr"

            for entry in listed:
                ok = runner.invoke(app, ["show", entry, "--vault", str(anchor_vault)])
                assert ok.exit_code == 0, (
                    f"Listed entry {entry!r} did not roundtrip through the parser: "
                    f"stderr={ok.stderr!r}"
                )

    def test_show_section_ambiguous_lists_with_line_numbers(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Same heading text twice in one note → both candidates listed with line numbers."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            result = runner.invoke(
                app, ["show", "dupes#Repeat", "--vault", str(anchor_vault)]
            )
            assert result.exit_code != 0
            assert "Ambiguous section" in result.stderr
            assert "dupes#Alpha#Repeat" in result.stderr
            assert "dupes#Beta#Repeat" in result.stderr
            # Line numbers ('L<n>') give the user something to anchor on
            assert re.search(r"L\d+:", result.stderr)

    def test_show_help_documents_anchor_and_ambiguity(self, runner: CliRunner):
        """`show --help` mentions `#Heading` syntax and the basename-collision behavior."""
        result = runner.invoke(app, ["show", "--help"])
        assert result.exit_code == 0
        assert "#Heading" in result.output
        assert "candidates" in result.output.lower()

    def test_show_section_collapses_hash_runs(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Runs of '#' collapse to a single separator so raw markdown heading
        prefixes (e.g. 'Note##Setup###Installation') resolve cleanly. The
        hash count signals heading level in markdown — we don't need it for
        matching, and rejecting it would break copy-paste from the source."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            single = runner.invoke(
                app, ["show", "nested#Setup#Installation", "--vault", str(anchor_vault)]
            )
            doubled = runner.invoke(
                app, ["show", "nested##Setup###Installation", "--vault", str(anchor_vault)]
            )
            assert single.exit_code == 0
            assert doubled.exit_code == 0
            assert single.output == doubled.output

    def test_show_trailing_hash_with_no_section_errors(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """`Note#` and `Note##` (no heading after the '#') error rather than
        silently returning the whole note — the hash signals 'I want to
        anchor somewhere' and an empty target is almost certainly a typo."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            for arg in ("nested#", "nested##"):
                result = runner.invoke(
                    app, ["show", arg, "--vault", str(anchor_vault)]
                )
                assert result.exit_code != 0, f"{arg!r} unexpectedly succeeded"
                assert "no heading" in result.stderr.lower()

    def test_show_section_trailing_newline_is_single(
        self, runner: CliRunner, anchor_vault: Path, configured_mock: Mock
    ):
        """Section output ends with exactly one trailing newline regardless of where in the file."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # 'Other' is the last section in nested.md; the source ends with `Other body.\n`,
            # so a naive join+echo combo would produce two trailing newlines.
            last = runner.invoke(
                app, ["show", "nested#Other", "--vault", str(anchor_vault)]
            )
            assert last.exit_code == 0
            assert last.output.endswith("Other body.\n")
            assert not last.output.endswith("Other body.\n\n")

            # And a mid-file section ending right before another heading.
            mid = runner.invoke(
                app, ["show", "nested#Setup#Configuration", "--vault", str(anchor_vault)]
            )
            assert mid.exit_code == 0
            assert mid.output.endswith("Config steps.\n")
            assert not mid.output.endswith("Config steps.\n\n")

    def test_show_section_unclosed_fence_suppresses_later_headings(
        self, runner: CliRunner, tmp_path: Path, configured_mock: Mock
    ):
        """An unclosed fence at EOF makes subsequent headings unreachable (documented behavior)."""
        vault = tmp_path / "unclosed-vault"
        vault.mkdir()
        (vault / "broken.md").write_text(
            "## Before\n"
            "Reachable body.\n"
            "\n"
            "```python\n"
            "# this fence is never closed\n"
            "## After\n"
            "After body.\n"
        )
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Heading before the unclosed fence still works.
            ok = runner.invoke(app, ["show", "broken#Before", "--vault", str(vault)])
            assert ok.exit_code == 0
            assert "Reachable body." in ok.output

            # Heading inside the unclosed fence is treated as code, not a section.
            missing = runner.invoke(
                app, ["show", "broken#After", "--vault", str(vault)]
            )
            assert missing.exit_code != 0
            assert "Section not found" in missing.stderr


class TestSuggestLinksCommand:
    """Test the suggest-links command."""

    def test_suggest_links_shows_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """suggest-links displays high-similarity unlinked pairs."""
        # Create vault files with wikilinks
        (vault_path / "note1.md").write_text("# Note One\n\nContent about [[note2]].\n")
        (vault_path / "note2.md").write_text("# Note Two\n\nDifferent content.\n")
        (vault_path / "note3.md").write_text("# Note Three\n\nMore content.\n")

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            # Index so the DB has data
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["suggest-links", "--vault", str(vault_path), "--threshold", "0.0"]
            )

            assert result.exit_code == 0
            # note1-note2 are linked, so they should NOT appear
            # But unlinked pairs should appear (at threshold 0.0 everything qualifies)
            assert "suggest" in result.output.lower() or "note" in result.output.lower()

    def test_suggest_links_no_results(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """suggest-links with high threshold returns no results gracefully."""
        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, ["suggest-links", "--vault", str(vault_path), "--threshold", "0.999"]
            )

            assert result.exit_code == 0
            assert "no" in result.output.lower()

    def test_suggest_links_help(self, runner: CliRunner):
        """suggest-links --help shows options."""
        result = runner.invoke(app, ["suggest-links", "--help"])

        assert result.exit_code == 0
        assert "--threshold" in result.output
        assert "--limit" in result.output
        assert "--exclude-same-folder" in result.output

    def test_suggest_links_exclude_same_folder(
        self, runner: CliRunner, vault_path: Path, configured_mock: Mock
    ):
        """--exclude-same-folder filters out pairs in the same folder."""
        sub = vault_path / "Daily Log"
        sub.mkdir()
        (sub / "day1.md").write_text("# Day 1\n\nSome content.\n")
        (sub / "day2.md").write_text("# Day 2\n\nSimilar content.\n")

        with patch("obsidian_semantic.cli.load_config", return_value=configured_mock):
            runner.invoke(app, ["index", "--vault", str(vault_path)])

            result = runner.invoke(
                app, [
                    "suggest-links", "--vault", str(vault_path),
                    "--threshold", "0.0",
                    "--exclude-same-folder", "Daily Log",
                ]
            )

            assert result.exit_code == 0


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
        assert "suggest-links" in result.output
        assert "show" in result.output

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

    def test_search_help_describes_folder_match(self, runner: CliRunner):
        """--folder help text clarifies it's a path prefix, case-sensitive."""
        result = runner.invoke(app, ["search", "--help"])

        assert result.exit_code == 0
        assert "path prefix" in result.output.lower()
        assert "case-sensitive" in result.output.lower()
