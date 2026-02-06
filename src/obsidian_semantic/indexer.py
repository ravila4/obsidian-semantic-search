"""Vault indexer - orchestrates crawling, chunking, and embedding."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from obsidian_semantic.chunker import Chunk, chunk_note, parse_note, strip_inline_tags
from obsidian_semantic.db import ChunkRecord, SemanticDB

if TYPE_CHECKING:
    from collections.abc import Iterator

    from obsidian_semantic.embedder.base import Embedder


@dataclass
class IndexResult:
    """Result of an indexing operation."""

    files_processed: int = 0
    files_deleted: int = 0
    chunks_created: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @property
    def duration_seconds(self) -> float:
        """Return duration in seconds."""
        if self.started_at and self.finished_at:
            return (self.finished_at - self.started_at).total_seconds()
        return 0.0


# Default patterns to ignore
DEFAULT_IGNORE_PATTERNS = [
    ".obsidian",
    ".git",
    ".trash",
    ".DS_Store",
]


def make_embedding_text(chunk: Chunk) -> str:
    """Create text for embedding that includes folder path, title, and headers.

    This improves search quality by ensuring queries like "python testing"
    can match notes titled "Unit Testing" in a Python folder, even if
    "python" doesn't appear in the chunk content.

    Inline tags are stripped from the content since they're stored separately
    in metadata. This prevents tag-heavy notes from dominating search results.

    Args:
        chunk: The chunk to create embedding text for.

    Returns:
        Text suitable for embedding, with path > title > headers > content.
    """
    # Extract folder context from file path (e.g., "Programming/Python" from
    # "Programming/Python/Unit Testing.md")
    path_parts = chunk.file_path.rsplit("/", 1)
    folder_context = path_parts[0] if len(path_parts) > 1 else ""

    # Build breadcrumb: Folder > Title > Header1 > Header2
    breadcrumb_parts = []
    if folder_context:
        breadcrumb_parts.append(folder_context.replace("/", " > "))
    breadcrumb_parts.append(chunk.title)
    breadcrumb_parts.extend(chunk.headers)
    header_path = " > ".join(breadcrumb_parts)

    # Strip inline tags from content - they're in metadata already
    content = strip_inline_tags(chunk.text).strip()

    # Combine with content
    if content:
        return f"{header_path}\n\n{content}"
    else:
        return header_path


class VaultIndexer:
    """Indexes an Obsidian vault for semantic search.

    Handles vault traversal, change detection, and coordination of
    chunking, embedding, and storage.
    """

    def __init__(
        self,
        vault_path: str | Path,
        db_path: str | Path,
        embedder: Embedder,
        ignore_patterns: list[str] | None = None,
    ):
        """Initialize the indexer.

        Args:
            vault_path: Path to the Obsidian vault root.
            db_path: Path to the LanceDB database directory.
            embedder: Embedder instance for generating vectors.
            ignore_patterns: Additional glob patterns to ignore.
        """
        self._vault_path = Path(vault_path)
        self._embedder = embedder
        self._ignore_patterns = DEFAULT_IGNORE_PATTERNS.copy()
        if ignore_patterns:
            self._ignore_patterns.extend(ignore_patterns)

        self._db = SemanticDB(db_path, dimension=embedder.dimension)

    def index(self, full: bool = False) -> IndexResult:
        """Index the vault.

        Args:
            full: If True, reindex all files. If False (default),
                only process new/modified files.

        Returns:
            IndexResult with statistics about the operation.
        """
        result = IndexResult(started_at=datetime.now())

        # Discover files
        files = list(self._discover_files())

        if full:
            # Process all files
            to_process = files
            to_delete: list[str] = []
        else:
            # Detect changes
            changes = self._detect_changes(files)
            to_process = changes["new"] + changes["modified"]
            to_delete = changes["deleted"]

        # Process files
        for file_path in to_process:
            try:
                self._index_file(file_path)
                result.files_processed += 1
            except Exception as e:
                result.errors.append(f"{file_path}: {e}")

        # Remove deleted files
        for rel_path in to_delete:
            self._db.delete_by_file(rel_path)
            result.files_deleted += 1

        # Update stats
        stats = self._db.get_stats()
        result.chunks_created = stats.chunk_count
        result.finished_at = datetime.now()

        return result

    def _discover_files(self) -> Iterator[Path]:
        """Discover all markdown files in the vault.

        Yields:
            Absolute paths to markdown files.
        """
        for path in self._vault_path.rglob("*.md"):
            # Check if any part of the path matches ignore patterns
            rel_path = path.relative_to(self._vault_path)
            if self._should_ignore(rel_path):
                continue
            yield path

    def _should_ignore(self, rel_path: Path) -> bool:
        """Check if a path should be ignored.

        Args:
            rel_path: Path relative to vault root.

        Returns:
            True if the path matches any ignore pattern.
        """
        path_str = str(rel_path)
        for pattern in self._ignore_patterns:
            # Check if any part of the path matches
            for part in rel_path.parts:
                if fnmatch.fnmatch(part, pattern):
                    return True
            # Also check the full path
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(rel_path.name, pattern):
                return True
        return False

    def _detect_changes(self, files: list[Path]) -> dict[str, list]:
        """Detect new, modified, and deleted files.

        Args:
            files: List of currently existing file paths.

        Returns:
            Dict with 'new', 'modified', and 'deleted' lists.
        """
        new_files: list[Path] = []
        modified_files: list[Path] = []

        # Track which files we've seen
        current_paths = set()

        for file_path in files:
            rel_path = str(file_path.relative_to(self._vault_path))
            current_paths.add(rel_path)

            # Check if file is in database
            meta = self._db.get_file_metadata(rel_path)

            if meta is None:
                new_files.append(file_path)
            else:
                # Check mtime
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > meta.indexed_at:
                    modified_files.append(file_path)

        # Find deleted files
        stats = self._db.get_stats()
        if stats.file_count > 0:
            # Get all indexed file paths
            all_indexed = self._get_all_indexed_paths()
            deleted_files = [p for p in all_indexed if p not in current_paths]
        else:
            deleted_files = []

        return {
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files,
        }

    def _get_all_indexed_paths(self) -> set[str]:
        """Get all file paths currently in the index.

        Returns:
            Set of relative file paths.
        """
        # This is a bit inefficient, but works for now
        # Could be optimized with a separate metadata table
        table = self._db._table()
        results = table.search().select(["file_path"]).limit(100000).to_list()
        return {r["file_path"] for r in results}

    def _index_file(self, file_path: Path) -> None:
        """Index a single file.

        Args:
            file_path: Absolute path to the file.
        """
        rel_path = str(file_path.relative_to(self._vault_path))
        content = file_path.read_text(encoding="utf-8")

        # Parse and chunk
        metadata, body = parse_note(content, rel_path)
        title = file_path.stem  # Use filename as title

        chunks = list(chunk_note(body, rel_path, title, metadata))

        if not chunks:
            return

        # Generate embeddings using text that includes title + headers
        texts = [make_embedding_text(chunk) for chunk in chunks]
        vectors = self._embedder.embed_document(texts)

        # Create records
        now = datetime.now()
        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

        records = [
            ChunkRecord(
                id=chunk.id,
                file_path=rel_path,
                title=chunk.title,
                headers=chunk.headers,
                text=chunk.text,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                tags=metadata.tags,
                created_at=metadata.created,
                modified_at=file_mtime,
                indexed_at=now,
                vector=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        # Delete old chunks for this file, then insert new ones
        self._db.delete_by_file(rel_path)
        self._db.upsert_chunks(records)
