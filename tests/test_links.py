"""Tests for the link suggestion module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from obsidian_semantic.links import (
    build_wikilink_graph,
    find_suggestions,
    get_note_embeddings,
)


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    """Create a small test vault with wikilinks."""
    vault = tmp_path / "vault"
    vault.mkdir()

    (vault / "alpha.md").write_text("# Alpha\n\nLinks to [[Beta]] and [[gamma]].\n")
    (vault / "beta.md").write_text("# Beta\n\nLinks back to [[Alpha]].\n")
    (vault / "gamma.md").write_text("# Gamma\n\nNo outgoing links.\n")
    (vault / "orphan.md").write_text("# Orphan\n\nNo links at all.\n")

    return vault


class TestBuildWikilinkGraph:
    """Test wikilink graph construction."""

    def test_bidirectional_links(self, vault: Path):
        """Links are stored bidirectionally."""
        graph = build_wikilink_graph(vault, ignore_patterns=[])

        # alpha -> beta (explicit) and beta -> alpha (explicit)
        assert "beta.md" in graph["alpha.md"]
        assert "alpha.md" in graph["beta.md"]

    def test_resolved_target(self, vault: Path):
        """[[gamma]] resolves to gamma.md even with different case."""
        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert "gamma.md" in graph["alpha.md"]
        # gamma has no outgoing links but should appear from the reverse edge
        assert "alpha.md" in graph["gamma.md"]

    def test_orphan_has_no_links(self, vault: Path):
        """A note with no wikilinks has an empty adjacency set."""
        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert graph["orphan.md"] == set()

    def test_ignore_patterns(self, vault: Path):
        """Files matching ignore patterns are excluded."""
        sub = vault / "Templates"
        sub.mkdir()
        (sub / "template.md").write_text("# Template\n\n[[Alpha]]\n")

        graph = build_wikilink_graph(vault, ignore_patterns=["Templates/*"])

        assert "Templates/template.md" not in graph

    def test_nested_folder(self, vault: Path):
        """Notes in subfolders resolve correctly."""
        sub = vault / "notes"
        sub.mkdir()
        (sub / "deep.md").write_text("# Deep\n\nLinks to [[Alpha]].\n")

        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert "alpha.md" in graph["notes/deep.md"]
        assert "notes/deep.md" in graph["alpha.md"]


class TestWikilinkResolution:
    """Test wikilink target resolution edge cases."""

    def test_aliased_link(self, vault: Path):
        """[[Note|alias]] resolves to the note, not the alias."""
        (vault / "linked.md").write_text("See [[alpha|my alias]].\n")

        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert "alpha.md" in graph["linked.md"]

    def test_section_link(self, vault: Path):
        """[[Note#Section]] strips the section and resolves."""
        (vault / "linked.md").write_text("See [[beta#intro]].\n")

        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert "beta.md" in graph["linked.md"]

    def test_unresolved_link(self, vault: Path):
        """Links to non-existent notes are silently ignored."""
        (vault / "linked.md").write_text("See [[nonexistent]].\n")

        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert graph["linked.md"] == set()

    def test_self_link_ignored(self, vault: Path):
        """A note linking to itself is not added to the graph."""
        (vault / "self.md").write_text("# Self\n\nSee [[self]].\n")

        graph = build_wikilink_graph(vault, ignore_patterns=[])

        assert "self.md" not in graph["self.md"]


class TestGetNoteEmbeddings:
    """Test per-note embedding averaging."""

    def test_averages_chunks(self, tmp_path: Path):
        """Multi-chunk note gets the mean of its chunk vectors."""
        from datetime import datetime
        from obsidian_semantic.db import ChunkRecord, SemanticDB

        db_path = tmp_path / "test_db"
        db = SemanticDB(db_path, dimension=4)

        now = datetime.now()
        db.upsert_chunks([
            ChunkRecord(
                id="a.md#0", file_path="a.md", title="A", headers=[], text="chunk0",
                start_line=1, end_line=5, tags=[], created_at=now,
                modified_at=now, indexed_at=now, vector=[1.0, 0.0, 0.0, 0.0],
            ),
            ChunkRecord(
                id="a.md#1", file_path="a.md", title="A", headers=[], text="chunk1",
                start_line=6, end_line=10, tags=[], created_at=now,
                modified_at=now, indexed_at=now, vector=[0.0, 1.0, 0.0, 0.0],
            ),
            ChunkRecord(
                id="b.md#0", file_path="b.md", title="B", headers=[], text="chunk",
                start_line=1, end_line=5, tags=[], created_at=now,
                modified_at=now, indexed_at=now, vector=[0.0, 0.0, 1.0, 0.0],
            ),
        ])

        embeddings = get_note_embeddings(db)

        assert set(embeddings.keys()) == {"a.md", "b.md"}
        np.testing.assert_array_almost_equal(
            embeddings["a.md"], [0.5, 0.5, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            embeddings["b.md"], [0.0, 0.0, 1.0, 0.0]
        )


class TestFindSuggestions:
    """Test the suggestion filtering and ranking logic."""

    @pytest.fixture
    def setup(self):
        """Create a simple 3-note scenario with known similarities."""
        files = ["a.md", "b.md", "folder/c.md"]
        # Build a 3x3 similarity matrix
        sim = np.array([
            [1.0,  0.9,  0.85],
            [0.9,  1.0,  0.5],
            [0.85, 0.5,  1.0],
        ])
        links: dict[str, set[str]] = {
            "a.md": {"b.md"},  # a and b are linked
            "b.md": {"a.md"},
        }
        return files, sim, links

    def test_filters_linked_pairs(self, setup):
        """Already-linked pairs are excluded."""
        files, sim, links = setup

        results = find_suggestions(files, sim, links, threshold=0.5, limit=10)

        # a-b is linked (sim=0.9) so it should NOT appear
        pairs = {(a, b) for a, b, _ in results}
        assert ("a.md", "b.md") not in pairs

    def test_returns_unlinked_above_threshold(self, setup):
        """Unlinked pairs above threshold appear in results."""
        files, sim, links = setup

        results = find_suggestions(files, sim, links, threshold=0.8, limit=10)

        # a-c (sim=0.85, unlinked) should appear
        pairs = {(a, b) for a, b, _ in results}
        assert ("a.md", "folder/c.md") in pairs

    def test_excludes_below_threshold(self, setup):
        """Pairs below threshold are excluded."""
        files, sim, links = setup

        results = find_suggestions(files, sim, links, threshold=0.8, limit=10)

        # b-c (sim=0.5) is below 0.8 threshold
        pairs = {(a, b) for a, b, _ in results}
        assert ("b.md", "folder/c.md") not in pairs

    def test_sorts_by_score_descending(self):
        """Output is ordered by similarity descending."""
        files = ["a.md", "b.md", "c.md", "d.md"]
        sim = np.array([
            [1.0,  0.95, 0.85, 0.90],
            [0.95, 1.0,  0.80, 0.88],
            [0.85, 0.80, 1.0,  0.82],
            [0.90, 0.88, 0.82, 1.0],
        ])
        links: dict[str, set[str]] = {}

        results = find_suggestions(files, sim, links, threshold=0.8, limit=10)
        scores = [score for _, _, score in results]

        assert scores == sorted(scores, reverse=True)

    def test_respects_limit(self):
        """Only returns up to limit results."""
        files = ["a.md", "b.md", "c.md"]
        sim = np.array([
            [1.0,  0.95, 0.90],
            [0.95, 1.0,  0.85],
            [0.90, 0.85, 1.0],
        ])
        links: dict[str, set[str]] = {}

        results = find_suggestions(files, sim, links, threshold=0.8, limit=1)

        assert len(results) == 1

    def test_exclude_same_folder(self, setup):
        """Pairs in an excluded folder are skipped."""
        # Put two notes in the same folder
        files = ["Daily Log/a.md", "Daily Log/b.md", "notes/c.md"]
        sim = np.array([
            [1.0,  0.95, 0.85],
            [0.95, 1.0,  0.80],
            [0.85, 0.80, 1.0],
        ])
        links: dict[str, set[str]] = {}

        results = find_suggestions(
            files, sim, links,
            threshold=0.8, limit=10,
            exclude_same_folder={"Daily Log"},
        )

        # a-b pair (both in Daily Log) should be excluded
        pairs = {(a, b) for a, b, _ in results}
        assert ("Daily Log/a.md", "Daily Log/b.md") not in pairs
        # a-c and b-c should still appear (different folders)
        assert ("Daily Log/a.md", "notes/c.md") in pairs
