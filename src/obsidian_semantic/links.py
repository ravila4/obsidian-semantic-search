"""Link analysis for suggesting missing wikilinks between notes."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from obsidian_semantic.chunker import WIKILINK_PATTERN
from obsidian_semantic.db import SemanticDB
from obsidian_semantic.indexer import should_ignore


def _build_stem_map(vault_path: Path, ignore_patterns: list[str]) -> dict[str, list[str]]:
    """Map note stem (lowercase) -> list of relative file paths."""
    stem_map: dict[str, list[str]] = {}
    for md in vault_path.rglob("*.md"):
        rel_path = md.relative_to(vault_path)
        if should_ignore(rel_path, ignore_patterns):
            continue
        stem = md.stem.lower()
        stem_map.setdefault(stem, []).append(str(rel_path))
    return stem_map


def _resolve_wikilink_target(
    target: str, stem_map: dict[str, list[str]]
) -> str | None:
    """Resolve a wikilink target to a relative file path (best effort).

    Handles [[Note]], [[folder/Note]], [[Note#Section]], [[Note|alias]].
    """
    # Strip section reference
    if "#" in target:
        target = target.split("#", 1)[0]
    target = target.strip()
    if not target:
        return None

    stem = target.rsplit("/", 1)[-1].lower()
    paths = stem_map.get(stem)
    if paths:
        return paths[0]
    return None


def build_wikilink_graph(
    vault_path: Path, ignore_patterns: list[str]
) -> dict[str, set[str]]:
    """Scan vault files, extract wikilinks, return bidirectional adjacency set.

    Args:
        vault_path: Root path of the Obsidian vault.
        ignore_patterns: Glob patterns for files to skip (e.g. ["Templates/*"]).

    Returns:
        Dict mapping relative file path -> set of linked file paths.
    """
    stem_map = _build_stem_map(vault_path, ignore_patterns)
    links: dict[str, set[str]] = {}

    for md in vault_path.rglob("*.md"):
        rel_path = md.relative_to(vault_path)
        if should_ignore(rel_path, ignore_patterns):
            continue

        rel = str(rel_path)

        try:
            content = md.read_text(encoding="utf-8")
        except Exception:
            continue

        links.setdefault(rel, set())

        for match in WIKILINK_PATTERN.finditer(content):
            target_text = match.group(1)
            resolved = _resolve_wikilink_target(target_text, stem_map)
            if resolved and resolved != rel:
                links[rel].add(resolved)
                links.setdefault(resolved, set()).add(rel)

    return links


def get_note_embeddings(db: SemanticDB) -> dict[str, np.ndarray]:
    """Pull all chunks from the DB and average vectors per note.

    Args:
        db: An open SemanticDB instance.

    Returns:
        Dict mapping relative file path -> averaged embedding vector.
    """
    vectors_by_file = db.get_all_vectors()
    return {fp: np.mean(vecs, axis=0) for fp, vecs in vectors_by_file.items()}


def _top_folder(path: str) -> str:
    """Get the top-level folder of a path, or empty string for root files."""
    parts = path.split("/", 1)
    return parts[0] if len(parts) > 1 else ""


def cosine_similarity_matrix(
    embeddings: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray]:
    """Compute NxN cosine similarity matrix.

    Args:
        embeddings: Dict mapping file path -> embedding vector.

    Returns:
        Tuple of (sorted file list, NxN similarity matrix).
    """
    files = sorted(embeddings.keys())
    matrix = np.array([embeddings[f] for f in files], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    sim = matrix @ matrix.T
    return files, sim


def find_suggestions(
    files: list[str],
    sim_matrix: np.ndarray,
    links: dict[str, set[str]],
    threshold: float,
    limit: int,
    exclude_same_folder: set[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Find unlinked high-similarity note pairs.

    Args:
        files: Ordered list of file paths (matching sim_matrix indices).
        sim_matrix: NxN cosine similarity matrix.
        links: Bidirectional adjacency set from build_wikilink_graph.
        threshold: Minimum similarity to include.
        limit: Maximum number of suggestions to return.
        exclude_same_folder: Skip pairs where both notes share a top folder
            in this set.

    Returns:
        List of (file_a, file_b, similarity) sorted by similarity descending.
    """
    exclude_same_folder = exclude_same_folder or set()
    n = len(files)
    suggestions: list[tuple[str, str, float]] = []

    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score < threshold:
                continue

            a, b = files[i], files[j]

            # Cheapest filter first: set membership
            if b in links.get(a, set()):
                continue

            if exclude_same_folder:
                folder_a = _top_folder(a)
                folder_b = _top_folder(b)
                if folder_a == folder_b and folder_a in exclude_same_folder:
                    continue

            suggestions.append((a, b, score))

    suggestions.sort(key=lambda x: x[2], reverse=True)
    return suggestions[:limit]
