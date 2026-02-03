"""Markdown parsing and chunking for Obsidian notes."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime

import yaml


@dataclass
class NoteMetadata:
    """Metadata extracted from note frontmatter and body."""

    tags: list[str] = field(default_factory=list)
    created: datetime | None = None
    modified: datetime | None = None
    aliases: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A semantic chunk of a note."""

    id: str  # "{file_path}#{header_slug}" or "{file_path}#chunk_{n}"
    file_path: str  # Relative to vault root
    title: str  # Note title (filename or H1)
    headers: list[str]  # Breadcrumb trail, e.g., ["Setup", "Installation"]
    text: str  # Chunk content
    start_line: int  # For linking back to source
    end_line: int


# Regex patterns
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
INLINE_TAG_PATTERN = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_-]*)")
WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Patterns for stripping code before tag extraction
FENCED_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_PATTERN = re.compile(r"`[^`]+`")

# Threshold for splitting on H3 instead of H2
MAX_SECTION_CHARS = 2000

# Minimum content length (excluding tags) to consider a note for chunking
MIN_NOTE_SIZE_FOR_CHUNKING = 200

# Minimum meaningful content for a chunk (excluding tags)
MIN_CHUNK_CONTENT = 30


def _content_length(text: str) -> int:
    """Get length of text excluding inline tags.

    Used to determine if a chunk has meaningful content beyond just tags.

    Args:
        text: The text to measure.

    Returns:
        Length of text after removing inline tags and stripping whitespace.
    """
    stripped = strip_inline_tags(text)
    return len(stripped.strip())


def strip_inline_tags(text: str) -> str:
    """Remove inline tags from text.

    Tags are stored separately in metadata, so removing them from
    embedding text prevents tag-heavy notes from dominating search
    results for generic queries.

    Args:
        text: Text that may contain inline tags like #python.

    Returns:
        Text with inline tags removed.
    """
    return INLINE_TAG_PATTERN.sub("", text)


def parse_note(content: str, file_path: str) -> tuple[NoteMetadata, str]:
    """Parse a note's frontmatter and body.

    Args:
        content: Raw markdown content of the note.
        file_path: Path to the note (for error messages).

    Returns:
        Tuple of (metadata, body) where body is the content without frontmatter.
    """
    metadata = NoteMetadata()
    body = content

    # Extract frontmatter
    match = FRONTMATTER_PATTERN.match(content)
    if match:
        frontmatter_text = match.group(1)
        body = content[match.end() :]

        try:
            frontmatter = yaml.safe_load(frontmatter_text)
            if frontmatter:
                metadata = _parse_frontmatter(frontmatter)
        except yaml.YAMLError:
            pass  # Invalid YAML, ignore frontmatter

    # Extract inline tags from body (excluding code blocks and inline code)
    body_without_code = _strip_code(body)
    inline_tags = INLINE_TAG_PATTERN.findall(body_without_code)
    for tag in inline_tags:
        if tag not in metadata.tags:
            metadata.tags.append(tag)

    return metadata, body


def _strip_code(text: str) -> str:
    """Remove fenced code blocks and inline code from text.

    Used to avoid extracting false tags from code content like `#define` or `#ff0000`.
    """
    # Remove fenced code blocks first (``` ... ```)
    text = FENCED_CODE_PATTERN.sub("", text)
    # Remove inline code (`...`)
    text = INLINE_CODE_PATTERN.sub("", text)
    return text


def _parse_frontmatter(data: dict) -> NoteMetadata:
    """Parse frontmatter dict into NoteMetadata."""
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [tags]

    aliases = data.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]

    created = None
    if "created" in data:
        created = _parse_date(data["created"])

    modified = None
    if "modified" in data:
        modified = _parse_date(data["modified"])

    return NoteMetadata(
        tags=list(tags),
        created=created,
        modified=modified,
        aliases=list(aliases),
    )


def _parse_date(value) -> datetime | None:
    """Parse a date from frontmatter (could be string, date, or datetime)."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "year"):  # date object
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        # Try common formats
        for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def chunk_note(
    body: str,
    file_path: str,
    title: str,
    metadata: NoteMetadata,
) -> Iterator[Chunk]:
    """Split a note body into semantic chunks.

    Short notes (under MIN_NOTE_SIZE_FOR_CHUNKING chars of content) are
    indexed as a single chunk to avoid creating tag-only or very short chunks.

    Larger notes are split on H2 headers. If a section exceeds MAX_SECTION_CHARS,
    it falls back to splitting on H3 headers within that section.

    Args:
        body: Note body (without frontmatter).
        file_path: Path to the note.
        title: Note title.
        metadata: Note metadata (unused here, but available for future use).

    Yields:
        Chunk objects for each section.
    """
    if not body.strip():
        return

    # Short notes: index as single chunk to avoid tag-only chunks
    if _content_length(body) < MIN_NOTE_SIZE_FOR_CHUNKING:
        text = _resolve_wikilinks(body.strip())
        yield Chunk(
            id=f"{file_path}#chunk_0",
            file_path=file_path,
            title=title,
            headers=[],
            text=text,
            start_line=1,
            end_line=len(body.split("\n")),
        )
        return

    lines = body.split("\n")
    sections = _split_into_sections(lines, level=2)

    chunk_num = 0
    for section in sections:
        # Check if section needs H3 splitting
        section_text = "\n".join(section["lines"])
        if len(section_text) > MAX_SECTION_CHARS and section["header"]:
            # Split this section on H3
            subsections = _split_into_sections(section["lines"], level=3)
            for subsection in subsections:
                chunk = _make_chunk(
                    subsection,
                    file_path,
                    title,
                    parent_headers=[section["header"]] if section["header"] else [],
                    chunk_num=chunk_num,
                )
                if chunk:
                    yield chunk
                    chunk_num += 1
        else:
            chunk = _make_chunk(
                section,
                file_path,
                title,
                parent_headers=[],
                chunk_num=chunk_num,
            )
            if chunk:
                yield chunk
                chunk_num += 1


def _split_into_sections(lines: list[str], level: int) -> list[dict]:
    """Split lines into sections based on header level.

    Args:
        lines: List of text lines.
        level: Header level to split on (2 for ##, 3 for ###).

    Returns:
        List of section dicts with keys: header, lines, start_line, end_line.
    """
    sections = []
    current_section = {
        "header": None,
        "lines": [],
        "start_line": 1,
        "end_line": 1,
    }

    header_prefix = "#" * level + " "

    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed

        # Check if this is a header at our target level
        stripped = line.strip()
        if stripped.startswith(header_prefix) and not stripped.startswith("#" * (level + 1)):
            # Save current section if it has content
            if current_section["lines"] or current_section["header"]:
                current_section["end_line"] = line_num - 1
                sections.append(current_section)

            # Start new section
            header_text = stripped[level + 1 :].strip()
            current_section = {
                "header": header_text,
                "lines": [],
                "start_line": line_num,
                "end_line": line_num,
            }
        else:
            current_section["lines"].append(line)

    # Don't forget the last section
    if current_section["lines"] or current_section["header"]:
        current_section["end_line"] = len(lines)
        sections.append(current_section)

    return sections


def _make_chunk(
    section: dict,
    file_path: str,
    title: str,
    parent_headers: list[str],
    chunk_num: int,
) -> Chunk | None:
    """Create a Chunk from a section dict."""
    text = "\n".join(section["lines"]).strip()

    # Resolve wikilinks
    text = _resolve_wikilinks(text)

    if not text and not section["header"]:
        return None

    # Skip chunks with only tags and no real content
    # (unless they have a header, which provides context)
    if not section["header"] and _content_length(text) < MIN_CHUNK_CONTENT:
        return None

    # Build headers breadcrumb
    headers = parent_headers.copy()
    if section["header"]:
        headers.append(section["header"])

    # Generate chunk ID
    if headers:
        slug = _slugify(headers[-1])
        chunk_id = f"{file_path}#{slug}"
    else:
        chunk_id = f"{file_path}#chunk_{chunk_num}"

    return Chunk(
        id=chunk_id,
        file_path=file_path,
        title=title,
        headers=headers,
        text=text,
        start_line=section["start_line"],
        end_line=section["end_line"],
    )


def _resolve_wikilinks(text: str) -> str:
    """Replace [[wikilinks]] with their display text.

    - [[Note]] -> Note
    - [[Note|alias]] -> alias
    - [[Note#Section]] -> Note > Section
    """

    def replace_link(match: re.Match) -> str:
        target = match.group(1)
        alias = match.group(2)

        if alias:
            return alias

        # Handle header links
        if "#" in target:
            parts = target.split("#", 1)
            return f"{parts[0]} > {parts[1]}" if parts[0] else parts[1]

        return target

    return WIKILINK_PATTERN.sub(replace_link, text)


def _slugify(text: str) -> str:
    """Convert header text to a URL-safe slug."""
    # Lowercase, replace spaces with hyphens, remove non-alphanumeric
    slug = text.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = slug.strip("-")
    return slug or "section"
