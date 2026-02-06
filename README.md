# obsidian-semantic

Semantic search for Obsidian vaults. Index your vault into vector embeddings, then search by meaning rather than keywords.

## Setup

```bash
uv sync
uv run obsidian-semantic configure
```

Configuration is stored in `~/.config/obsidian-semantic/config.yaml`. Supports Ollama (local) and Gemini embedders.

## Usage

### Index your vault

```bash
obsidian-semantic index                # incremental (new/modified files only)
obsidian-semantic index --full         # reindex everything
```

### Search

```bash
obsidian-semantic search "dependency injection patterns"
obsidian-semantic search "python testing" --limit 5
obsidian-semantic search "docker" --folder "Programming/"
obsidian-semantic search "habits" --tag "review"
```

### Find related notes

Find notes similar to a given note, useful for discovering connections, linking, or deduplication.

```bash
obsidian-semantic related "Programming/Python/Unit Testing.md"
obsidian-semantic related "Daily/2026-02-05.md" --limit 5
```

Works with both indexed and unindexed notes -- if the note isn't in the index yet, it gets chunked and embedded on the fly.

### Status

```bash
obsidian-semantic status
```

### Options

All commands accept `--vault <path>` to specify the vault. Alternatively, set `OBSIDIAN_VAULT` or configure a default with `obsidian-semantic configure --vault <path>`.

## Embedding Backends

Configuration lives in `~/.config/obsidian-semantic/config.yaml`. You can also place a `.obsidian-semantic.yaml` in your vault root to override per-vault.

After changing the embedder or model, reindex with `obsidian-semantic index --full`.

### Ollama with Nomic (default)

Local embeddings with [nomic-embed-text](https://ollama.com/library/nomic-embed-text) (768 dimensions). Uses `search_query:`/`search_document:` prefixes for asymmetric retrieval.

```yaml
vault: ~/Documents/Obsidian-Notes
embedder:
  type: ollama
  model: nomic-embed-text
  dimension: 768
  query_prefix: "search_query: "
  document_prefix: "search_document: "
```

```bash
ollama pull nomic-embed-text
```

### Ollama with Qwen3-embedding

Higher-quality embeddings with [qwen3-embedding](https://ollama.com/library/qwen3-embedding) (4096 dimensions). Uses an instruction prefix for queries to improve retrieval.

```yaml
vault: ~/Documents/Obsidian-Notes
embedder:
  type: ollama
  model: qwen3-embedding:8b
  dimension: 4096
  query_prefix: "Instruct: Given a search query, retrieve relevant notes\nQuery: "
```

```bash
ollama pull qwen3-embedding:8b
```

### Gemini

Cloud embeddings via Google's [gemini-embedding-001](https://ai.google.dev/gemini-api/docs/embeddings) (3072 dimensions). Handles query vs. document task types automatically -- no prefix config needed. Requires a `GEMINI_API_KEY` environment variable.

```yaml
vault: ~/Documents/Obsidian-Notes
embedder:
  type: gemini
  model: gemini-embedding-001
  dimension: 3072
```

### Advanced Options

**Timeout Configuration**

The embedder request timeout (default: 30 seconds) can be increased for large files or slower models:

```yaml
embedder:
  timeout: 60.0  # seconds
```

If you see timeout errors during indexing, try increasing this value. Very large notes with extensive JSON or code blocks may need 60-120 seconds.
