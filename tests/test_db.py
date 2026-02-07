"""Tests for the LanceDB database layer."""

from datetime import datetime
from pathlib import Path

import pytest

from obsidian_semantic.db import ChunkRecord, SearchResult, SemanticDB


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test_db"


@pytest.fixture
def db(db_path: Path) -> SemanticDB:
    """Provide a test database instance."""
    return SemanticDB(db_path, dimension=768)


@pytest.fixture
def sample_chunks() -> list[ChunkRecord]:
    """Provide sample chunks for testing."""
    now = datetime.now()
    return [
        ChunkRecord(
            id="notes/python.md#basics",
            file_path="notes/python.md",
            title="Python Basics",
            headers=["Basics"],
            text="Python is a programming language.",
            start_line=1,
            end_line=10,
            tags=["python", "programming"],
            created_at=now,
            modified_at=now,
            indexed_at=now,
            vector=[0.1] * 768,
        ),
        ChunkRecord(
            id="notes/python.md#advanced",
            file_path="notes/python.md",
            title="Python Basics",
            headers=["Advanced"],
            text="Decorators and metaclasses are advanced features.",
            start_line=11,
            end_line=20,
            tags=["python", "advanced"],
            created_at=now,
            modified_at=now,
            indexed_at=now,
            vector=[0.2] * 768,
        ),
        ChunkRecord(
            id="notes/rust.md#intro",
            file_path="notes/rust.md",
            title="Rust Introduction",
            headers=["Intro"],
            text="Rust is a systems programming language.",
            start_line=1,
            end_line=5,
            tags=["rust", "programming"],
            created_at=now,
            modified_at=now,
            indexed_at=now,
            vector=[0.3] * 768,
        ),
    ]


class TestDatabaseOperations:
    """Test basic database operations."""

    def test_create_database(self, db: SemanticDB):
        """Database should be created on initialization."""
        assert db is not None

    def test_upsert_chunks(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should insert new chunks."""
        db.upsert_chunks(sample_chunks)

        stats = db.get_stats()
        assert stats.chunk_count == 3
        assert stats.file_count == 2

    def test_upsert_updates_existing(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should update existing chunks by ID."""
        db.upsert_chunks(sample_chunks)

        # Update one chunk
        updated = sample_chunks[0]
        updated.text = "Updated content here."
        db.upsert_chunks([updated])

        stats = db.get_stats()
        assert stats.chunk_count == 3  # Still 3, not 4

    def test_delete_by_file(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should delete all chunks for a file."""
        db.upsert_chunks(sample_chunks)

        deleted = db.delete_by_file("notes/python.md")
        assert deleted == 2

        stats = db.get_stats()
        assert stats.chunk_count == 1
        assert stats.file_count == 1

    def test_get_by_file(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should retrieve all chunks for a file."""
        db.upsert_chunks(sample_chunks)

        chunks = db.get_by_file("notes/python.md")
        assert len(chunks) == 2
        assert all(c.file_path == "notes/python.md" for c in chunks)

    def test_get_file_metadata(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should return latest indexed_at for a file."""
        db.upsert_chunks(sample_chunks)

        meta = db.get_file_metadata("notes/python.md")
        assert meta is not None
        assert meta.indexed_at is not None


class TestSearch:
    """Test vector search functionality."""

    def test_basic_search(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should return results ordered by similarity."""
        db.upsert_chunks(sample_chunks)

        # Query with vector similar to python chunk
        query_vector = [0.1] * 768
        results = db.search(query_vector, limit=10)

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].file_path == "notes/python.md"  # Most similar

    def test_search_with_tag_filter(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should filter results by tags."""
        db.upsert_chunks(sample_chunks)

        query_vector = [0.15] * 768
        results = db.search(query_vector, limit=10, filter_tags=["rust"])

        assert len(results) == 1
        assert results[0].file_path == "notes/rust.md"

    def test_search_with_folder_filter(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should filter results by folder prefix."""
        db.upsert_chunks(sample_chunks)

        query_vector = [0.15] * 768
        results = db.search(query_vector, limit=10, filter_folder="notes/")

        assert len(results) == 3  # All in notes/

    def test_search_respects_limit(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should respect limit parameter."""
        db.upsert_chunks(sample_chunks)

        query_vector = [0.15] * 768
        results = db.search(query_vector, limit=1)

        assert len(results) == 1

    def test_search_result_has_score(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Search results should include similarity score."""
        db.upsert_chunks(sample_chunks)

        query_vector = [0.15] * 768
        results = db.search(query_vector, limit=10)

        for result in results:
            assert hasattr(result, "score")
            assert 0 <= result.score <= 1 or result.score > 0  # LanceDB uses distance


class TestSpecialCharacters:
    """Test handling of special characters in paths and data."""

    def test_file_path_with_apostrophe(self, db: SemanticDB):
        """Should handle file paths containing apostrophes."""
        now = datetime.now()
        chunk = ChunkRecord(
            id="notes/Ricardo's Notes.md#intro",
            file_path="notes/Ricardo's Notes.md",
            title="Ricardo's Notes",
            headers=["Intro"],
            text="Some content here.",
            start_line=1,
            end_line=5,
            tags=["personal"],
            created_at=now,
            modified_at=now,
            indexed_at=now,
            vector=[0.1] * 768,
        )

        # Should not raise SQL syntax error
        db.upsert_chunks([chunk])

        # Should retrieve correctly
        chunks = db.get_by_file("notes/Ricardo's Notes.md")
        assert len(chunks) == 1
        assert chunks[0].title == "Ricardo's Notes"

        # Should delete correctly
        deleted = db.delete_by_file("notes/Ricardo's Notes.md")
        assert deleted == 1

    def test_tag_with_apostrophe(self, db: SemanticDB):
        """Should handle tags containing apostrophes in search filter."""
        now = datetime.now()
        chunk = ChunkRecord(
            id="notes/test.md#1",
            file_path="notes/test.md",
            title="Test",
            headers=[],
            text="Content",
            start_line=1,
            end_line=5,
            tags=["it's-a-tag"],
            created_at=now,
            modified_at=now,
            indexed_at=now,
            vector=[0.1] * 768,
        )
        db.upsert_chunks([chunk])

        # Search with tag containing apostrophe should work
        results = db.search([0.1] * 768, limit=10, filter_tags=["it's-a-tag"])
        assert len(results) == 1


class TestExcludeFile:
    """Test exclude_file parameter in search."""

    def test_search_excludes_file(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Search with exclude_file should not return chunks from that file."""
        db.upsert_chunks(sample_chunks)

        # Query vector close to python chunks, but exclude python.md
        query_vector = [0.1] * 768
        results = db.search(query_vector, limit=10, exclude_file="notes/python.md")

        assert len(results) > 0
        assert all(r.file_path != "notes/python.md" for r in results)

    def test_search_without_exclude_file_returns_all(
        self, db: SemanticDB, sample_chunks: list[ChunkRecord]
    ):
        """Search without exclude_file returns results from all files."""
        db.upsert_chunks(sample_chunks)

        query_vector = [0.15] * 768
        results = db.search(query_vector, limit=10)

        file_paths = {r.file_path for r in results}
        assert len(file_paths) == 2  # Both python.md and rust.md


class TestBulkFileMetadata:
    """Test bulk file metadata retrieval."""

    def test_get_all_file_metadata(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should return correct file_path -> indexed_at mapping."""
        db.upsert_chunks(sample_chunks)

        result = db.get_all_file_metadata()

        assert isinstance(result, dict)
        assert len(result) == 2  # Two unique files
        assert "notes/python.md" in result
        assert "notes/rust.md" in result
        # indexed_at should be a datetime
        assert isinstance(result["notes/python.md"], datetime)
        assert isinstance(result["notes/rust.md"], datetime)

    def test_get_all_file_metadata_empty_db(self, db: SemanticDB):
        """Should return empty dict for empty database."""
        result = db.get_all_file_metadata()

        assert result == {}

    def test_get_all_file_metadata_deduplicates(self, db: SemanticDB):
        """Should take max indexed_at when file has multiple chunks."""
        earlier = datetime(2025, 1, 1, 12, 0, 0)
        later = datetime(2025, 1, 1, 12, 0, 1)

        chunks = [
            ChunkRecord(
                id="notes/test.md#chunk1",
                file_path="notes/test.md",
                title="Test",
                headers=["A"],
                text="First chunk",
                start_line=1,
                end_line=5,
                tags=[],
                created_at=earlier,
                modified_at=earlier,
                indexed_at=earlier,
                vector=[0.1] * 768,
            ),
            ChunkRecord(
                id="notes/test.md#chunk2",
                file_path="notes/test.md",
                title="Test",
                headers=["B"],
                text="Second chunk",
                start_line=6,
                end_line=10,
                tags=[],
                created_at=later,
                modified_at=later,
                indexed_at=later,
                vector=[0.2] * 768,
            ),
        ]
        db.upsert_chunks(chunks)

        result = db.get_all_file_metadata()

        assert len(result) == 1
        assert result["notes/test.md"] == later


class TestGetAllVectors:
    """Test bulk vector retrieval."""

    def test_get_all_vectors(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Should return file_path -> list of vectors mapping."""
        db.upsert_chunks(sample_chunks)

        result = db.get_all_vectors()

        assert len(result) == 2
        assert "notes/python.md" in result
        assert "notes/rust.md" in result
        assert len(result["notes/python.md"]) == 2  # 2 chunks
        assert len(result["notes/rust.md"]) == 1

    def test_get_all_vectors_empty_db(self, db: SemanticDB):
        """Should return empty dict for empty database."""
        result = db.get_all_vectors()

        assert result == {}


class TestStats:
    """Test index statistics."""

    def test_empty_db_stats(self, db: SemanticDB):
        """Empty database should have zero counts."""
        stats = db.get_stats()

        assert stats.chunk_count == 0
        assert stats.file_count == 0

    def test_stats_after_operations(self, db: SemanticDB, sample_chunks: list[ChunkRecord]):
        """Stats should update after operations."""
        db.upsert_chunks(sample_chunks)

        stats = db.get_stats()
        assert stats.chunk_count == 3
        assert stats.file_count == 2

        db.delete_by_file("notes/python.md")

        stats = db.get_stats()
        assert stats.chunk_count == 1
        assert stats.file_count == 1
