"""Vault indexer - orchestrates crawling, chunking, and embedding."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
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
    files_skipped: int = 0
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


@dataclass
class PendingChanges:
    """Summary of files that need indexing."""

    new_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Return True if there are any pending changes."""
        return bool(self.new_files or self.modified_files or self.deleted_files)

    @property
    def total_count(self) -> int:
        """Return total number of pending changes."""
        return len(self.new_files) + len(self.modified_files) + len(self.deleted_files)


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

    def index(
        self,
        full: bool = False,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> IndexResult:
        """Index the vault.

        Args:
            full: If True, reindex all files. If False (default),
                only process new/modified files.
            progress_callback: Optional callback function called for each file.
                Receives (current, total, file_path) arguments.

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
        total_files = len(to_process)
        for idx, file_path in enumerate(to_process, start=1):
            rel_path = str(file_path.relative_to(self._vault_path))
            if progress_callback:
                progress_callback(idx, total_files, rel_path)

            try:
                indexed = self._index_file(file_path)
                if indexed:
                    result.files_processed += 1
                else:
                    result.files_skipped += 1
            except Exception as e:
                result.errors.append(f"{rel_path}: {e}")

        # Remove deleted files
        for rel_path in to_delete:
            self._db.delete_by_file(rel_path)
            result.files_deleted += 1

        # Update stats
        stats = self._db.get_stats()
        result.chunks_created = stats.chunk_count
        result.finished_at = datetime.now()

        return result

    def get_pending_changes(self) -> PendingChanges:
        """Get summary of files that need indexing.

        Filters out files with no indexable content (e.g., frontmatter-only stubs).

        Returns:
            PendingChanges with lists of new, modified, and deleted files.
        """
        files = list(self._discover_files())
        changes = self._detect_changes(files)

        return PendingChanges(
            new_files=[
                str(f.relative_to(self._vault_path))
                for f in changes["new"]
                if self._has_indexable_content(f)
            ],
            modified_files=[
                str(f.relative_to(self._vault_path))
                for f in changes["modified"]
                if self._has_indexable_content(f)
            ],
            deleted_files=changes["deleted"],
        )

    def _has_indexable_content(self, file_path: Path) -> bool:
        """Check if a file has content beyond frontmatter.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if the file has body content that would produce chunks.
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            _, body = parse_note(content, str(file_path))
            return bool(body.strip())
        except Exception:
            return False

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
        # Single bulk query instead of per-file lookups
        indexed = self._db.get_all_file_metadata()

        new_files: list[Path] = []
        modified_files: list[Path] = []
        current_paths = set()

        for file_path in files:
            rel_path = str(file_path.relative_to(self._vault_path))
            current_paths.add(rel_path)

            indexed_at = indexed.get(rel_path)
            if indexed_at is None:
                new_files.append(file_path)
            else:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime > indexed_at:
                    modified_files.append(file_path)

        # Find deleted files
        deleted_files = [p for p in indexed if p not in current_paths]

        return {
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files,
        }

    def _index_file(self, file_path: Path) -> bool:
        """Index a single file.

        Args:
            file_path: Absolute path to the file.

        Returns:
            True if chunks were created, False if file had no indexable content.
        """
        rel_path = str(file_path.relative_to(self._vault_path))
        content = file_path.read_text(encoding="utf-8")

        # Parse and chunk
        metadata, body = parse_note(content, rel_path)
        title = file_path.stem  # Use filename as title

        chunks = list(chunk_note(body, rel_path, title, metadata))

        if not chunks:
            return False

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
        return True
