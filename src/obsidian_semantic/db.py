"""LanceDB database layer for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa


@dataclass
class ChunkRecord:
    """A chunk record for storage in the database."""

    id: str  # Primary key: "{file_path}#{chunk_id}"
    file_path: str  # Relative path from vault root
    title: str  # Note title
    headers: list[str]  # Header breadcrumb
    text: str  # Chunk content
    start_line: int  # Source location
    end_line: int

    # Metadata for filtering
    tags: list[str]  # Combined frontmatter + inline tags
    created_at: datetime | None  # From frontmatter or file creation time
    modified_at: datetime  # From frontmatter or file mtime
    indexed_at: datetime  # When this chunk was last indexed

    # Embedding
    vector: list[float]  # Embedding vector


@dataclass
class SearchResult:
    """A search result from the database."""

    id: str
    file_path: str
    title: str
    headers: list[str]
    text: str
    score: float
    start_line: int


@dataclass
class FileMetadata:
    """Metadata about an indexed file."""

    file_path: str
    indexed_at: datetime


@dataclass
class IndexStats:
    """Statistics about the index."""

    chunk_count: int
    file_count: int
    last_indexed: datetime | None = None


CHUNKS_TABLE = "chunks"


def _escape_sql_string(value: str) -> str:
    """Escape single quotes for SQL string literals."""
    return value.replace("'", "''")


class SemanticDB:
    """LanceDB-backed semantic search database."""

    def __init__(self, db_path: str | Path, dimension: int = 768):
        """Initialize the database.

        Args:
            db_path: Path to the LanceDB directory.
            dimension: Embedding vector dimension.
        """
        self.db_path = Path(db_path)
        self.dimension = dimension
        self._db = lancedb.connect(str(self.db_path))
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Ensure the chunks table exists with correct schema."""
        existing_tables = self._db.list_tables().tables
        if CHUNKS_TABLE not in existing_tables:
            schema = self._get_schema()
            self._db.create_table(CHUNKS_TABLE, schema=schema)

    def _get_schema(self) -> pa.Schema:
        """Get the PyArrow schema for the chunks table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("file_path", pa.string()),
            pa.field("title", pa.string()),
            pa.field("headers", pa.list_(pa.string())),
            pa.field("text", pa.string()),
            pa.field("start_line", pa.int32()),
            pa.field("end_line", pa.int32()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("created_at", pa.timestamp("us")),
            pa.field("modified_at", pa.timestamp("us")),
            pa.field("indexed_at", pa.timestamp("us")),
            pa.field("vector", pa.list_(pa.float32(), self.dimension)),
        ])

    def _table(self) -> lancedb.table.Table:
        """Get the chunks table."""
        return self._db.open_table(CHUNKS_TABLE)

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        """Insert or update chunks by ID.

        Args:
            chunks: List of chunk records to upsert.
        """
        if not chunks:
            return

        table = self._table()

        # Convert to dicts for LanceDB
        data = [self._chunk_to_dict(chunk) for chunk in chunks]

        # Delete existing entries with same IDs, then add new ones
        ids = [_escape_sql_string(chunk.id) for chunk in chunks]
        table.delete(f"id IN {tuple(ids)!r}" if len(ids) > 1 else f"id = '{ids[0]}'")

        table.add(data)

    def _chunk_to_dict(self, chunk: ChunkRecord) -> dict[str, Any]:
        """Convert a ChunkRecord to a dict for storage."""
        return {
            "id": chunk.id,
            "file_path": chunk.file_path,
            "title": chunk.title,
            "headers": chunk.headers,
            "text": chunk.text,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "tags": chunk.tags,
            "created_at": chunk.created_at,
            "modified_at": chunk.modified_at,
            "indexed_at": chunk.indexed_at,
            "vector": chunk.vector,
        }

    def delete_by_file(self, file_path: str) -> int:
        """Remove all chunks for a file.

        Args:
            file_path: Path of the file to delete chunks for.

        Returns:
            Number of chunks deleted.
        """
        table = self._table()

        # Count before delete
        try:
            count_before = table.count_rows(f"file_path = '{_escape_sql_string(file_path)}'")
        except Exception:
            count_before = 0

        table.delete(f"file_path = '{_escape_sql_string(file_path)}'")

        return count_before

    def get_by_file(self, file_path: str) -> list[ChunkRecord]:
        """Get all chunks for a file.

        Args:
            file_path: Path of the file.

        Returns:
            List of chunk records.
        """
        table = self._table()

        try:
            escaped_path = _escape_sql_string(file_path)
            results = table.search().where(
                f"file_path = '{escaped_path}'", prefilter=True
            ).to_list()
            return [self._dict_to_chunk(r) for r in results]
        except Exception:
            return []

    def _dict_to_chunk(self, data: dict) -> ChunkRecord:
        """Convert a dict from the database to a ChunkRecord."""
        return ChunkRecord(
            id=data["id"],
            file_path=data["file_path"],
            title=data["title"],
            headers=data.get("headers", []),
            text=data["text"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            modified_at=data["modified_at"],
            indexed_at=data["indexed_at"],
            vector=data["vector"],
        )

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter_tags: list[str] | None = None,
        filter_folder: str | None = None,
        filter_after: datetime | None = None,
        exclude_file: str | None = None,
    ) -> list[SearchResult]:
        """Search for similar chunks.

        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results.
            filter_tags: Only return chunks with any of these tags.
            filter_folder: Only return chunks from files in this folder.
            filter_after: Only return chunks modified after this date.
            exclude_file: Exclude chunks from this file path.

        Returns:
            List of search results ordered by similarity.
        """
        table = self._table()

        # Build query
        query = table.search(query_vector).limit(limit)

        # Apply filters
        filters = []
        if filter_tags:
            # Check if any tag matches
            tag_conditions = " OR ".join(
                f"array_contains(tags, '{_escape_sql_string(tag)}')"
                for tag in filter_tags
            )
            filters.append(f"({tag_conditions})")
        if filter_folder:
            filters.append(f"starts_with(file_path, '{_escape_sql_string(filter_folder)}')")
        if filter_after:
            filters.append(f"modified_at > '{filter_after.isoformat()}'")
        if exclude_file:
            filters.append(f"file_path != '{_escape_sql_string(exclude_file)}'")

        if filters:
            query = query.where(" AND ".join(filters), prefilter=True)

        try:
            results = query.to_list()
        except Exception:
            return []

        return [
            SearchResult(
                id=r["id"],
                file_path=r["file_path"],
                title=r["title"],
                headers=r.get("headers", []),
                text=r["text"],
                score=1.0 / (1.0 + r.get("_distance", 0)),  # Convert distance to similarity
                start_line=r["start_line"],
            )
            for r in results
        ]

    def get_file_metadata(self, file_path: str) -> FileMetadata | None:
        """Get metadata about an indexed file.

        Args:
            file_path: Path of the file.

        Returns:
            FileMetadata if file is indexed, None otherwise.
        """
        table = self._table()

        try:
            results = (
                table.search()
                .where(f"file_path = '{_escape_sql_string(file_path)}'", prefilter=True)
                .limit(1)
                .to_list()
            )
            if results:
                return FileMetadata(
                    file_path=file_path,
                    indexed_at=results[0]["indexed_at"],
                )
        except Exception:
            pass

        return None

    def get_all_file_metadata(self) -> dict[str, datetime]:
        """Get metadata for all indexed files in a single query.

        Returns:
            Dict mapping file_path -> indexed_at (max across chunks).
        """
        table = self._table()

        try:
            results = table.search().select(["file_path", "indexed_at"]).limit(100000).to_list()
        except Exception:
            return {}

        if not results:
            return {}

        # Deduplicate: multiple chunks per file, take max indexed_at
        metadata: dict[str, datetime] = {}
        for r in results:
            path = r["file_path"]
            indexed_at = r["indexed_at"]
            if path not in metadata or indexed_at > metadata[path]:
                metadata[path] = indexed_at

        return metadata

    def get_all_vectors(self) -> dict[str, list[list[float]]]:
        """Get all embedding vectors grouped by file path.

        Returns:
            Dict mapping file_path -> list of chunk vectors.
        """
        table = self._table()

        try:
            results = table.search().select(["file_path", "vector"]).limit(100_000).to_list()
        except Exception:
            return {}

        if not results:
            return {}

        vectors_by_file: dict[str, list[list[float]]] = {}
        for r in results:
            vectors_by_file.setdefault(r["file_path"], []).append(r["vector"])

        return vectors_by_file

    def get_stats(self) -> IndexStats:
        """Get index statistics.

        Returns:
            IndexStats with chunk count, file count, etc.
        """
        table = self._table()

        try:
            chunk_count = table.count_rows()
            if chunk_count == 0:
                return IndexStats(chunk_count=0, file_count=0)

            # Get unique file paths
            results = table.search().select(["file_path", "indexed_at"]).limit(100000).to_list()
            file_paths = {r["file_path"] for r in results}
            file_count = len(file_paths)

            # Get max indexed_at
            last_indexed = max((r["indexed_at"] for r in results), default=None)

            return IndexStats(
                chunk_count=chunk_count,
                file_count=file_count,
                last_indexed=last_indexed,
            )
        except Exception:
            return IndexStats(chunk_count=0, file_count=0)
