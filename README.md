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
