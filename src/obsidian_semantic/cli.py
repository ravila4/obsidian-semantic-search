"""CLI for obsidian-semantic search."""

# ruff: noqa: B008  # Typer uses function calls in defaults (typer.Option) by design

from __future__ import annotations

import os
from pathlib import Path

import typer
import yaml

from obsidian_semantic.config import load_config
from obsidian_semantic.db import SemanticDB
from obsidian_semantic.indexer import VaultIndexer

app = typer.Typer(
    name="obsidian-semantic",
    help="Semantic search for Obsidian vaults.",
    no_args_is_help=True,
)

CONFIG_DIR = Path.home() / ".config" / "obsidian-semantic"


def _get_vault_path(vault: Path | None) -> Path:
    """Resolve vault path from arg, environment, or config file."""
    if vault:
        return vault
    env_vault = os.environ.get("OBSIDIAN_VAULT")
    if env_vault:
        return Path(env_vault)
    # Check config file
    config_file = CONFIG_DIR / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            data = yaml.safe_load(f)
            if data and "vault" in data:
                return Path(data["vault"])
    raise typer.BadParameter(
        "Vault path required. Use --vault, set OBSIDIAN_VAULT, or run 'configure --vault'."
    )


def _get_db_path(config_db: str, vault_path: Path) -> Path:
    """Resolve database path, making it relative to vault if needed."""
    db_path = Path(config_db)
    if not db_path.is_absolute():
        return vault_path / db_path
    return db_path


@app.command()
def status(
    vault: Path | None = typer.Option(
        None, "--vault", "-v", help="Path to Obsidian vault."
    ),
) -> None:
    """Show index status and statistics."""
    vault_path = _get_vault_path(vault)
    config = load_config(vault_path)
    db_path = _get_db_path(config.database, vault_path)

    embedder = config.create_embedder()

    db = SemanticDB(db_path, dimension=embedder.dimension)
    stats = db.get_stats()

    typer.echo(f"Vault: {vault_path}")
    typer.echo(f"Database: {db_path}")
    typer.echo(f"Indexed: {stats.file_count} files, {stats.chunk_count} chunks")
    if stats.last_indexed:
        typer.echo(f"Last indexed: {stats.last_indexed.strftime('%Y-%m-%d %H:%M:%S')}")


@app.command()
def index(
    full: bool = typer.Option(False, "--full", "-f", help="Reindex all files."),
    vault: Path | None = typer.Option(
        None, "--vault", "-v", help="Path to Obsidian vault."
    ),
) -> None:
    """Index the Obsidian vault."""
    vault_path = _get_vault_path(vault)
    config = load_config(vault_path)
    db_path = _get_db_path(config.database, vault_path)

    embedder = config.create_embedder()

    indexer = VaultIndexer(
        vault_path=vault_path,
        db_path=db_path,
        embedder=embedder,
        ignore_patterns=config.ignore,
    )

    mode = "full" if full else "incremental"
    typer.echo(f"Indexing vault ({mode})...")

    result = indexer.index(full=full)

    typer.echo(f"Processed {result.files_processed} files")
    if result.files_deleted:
        typer.echo(f"Removed {result.files_deleted} deleted files")
    if result.errors:
        typer.echo(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            typer.echo(f"  - {error}")
    typer.echo(f"Total indexed: {result.chunks_created} chunks")
    typer.echo(f"Duration: {result.duration_seconds:.2f}s")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results."),
    tags: list[str] | None = typer.Option(None, "--tag", "-t", help="Filter by tags."),
    folder: str | None = typer.Option(None, "--folder", help="Filter by folder."),
    vault: Path | None = typer.Option(
        None, "--vault", "-v", help="Path to Obsidian vault."
    ),
) -> None:
    """Search indexed content semantically."""
    vault_path = _get_vault_path(vault)
    config = load_config(vault_path)
    db_path = _get_db_path(config.database, vault_path)

    embedder = config.create_embedder()

    db = SemanticDB(db_path, dimension=embedder.dimension)

    # Generate query embedding
    query_vector = embedder.embed([query])[0]

    results = db.search(
        query_vector=query_vector,
        limit=limit,
        filter_tags=tags,
        filter_folder=folder,
    )

    if not results:
        typer.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        typer.echo(f"\n--- Result {i} (score: {result.score:.3f}) ---")
        typer.echo(f"File: {result.file_path}:{result.start_line}")
        if result.headers:
            typer.echo(f"Section: {' > '.join(result.headers)}")
        typer.echo(f"\n{result.text[:500]}{'...' if len(result.text) > 500 else ''}")


@app.command()
def configure(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration."),
    embedder: str | None = typer.Option(None, "--embedder", "-e", help="Embedder (ollama/gemini)."),
    model: str | None = typer.Option(None, "--model", "-m", help="Model name."),
    vault: Path | None = typer.Option(None, "--vault", "-v", help="Default vault path."),
) -> None:
    """Configure obsidian-semantic settings."""
    config_file = CONFIG_DIR / "config.yaml"

    if show:
        if config_file.exists():
            typer.echo(f"Config file: {config_file}")
            typer.echo(config_file.read_text())
        else:
            typer.echo(f"No config file found at {config_file}")
            typer.echo("Using defaults.")
        return

    # Load existing config to merge with
    config_data: dict = {}
    if config_file.exists():
        with open(config_file) as f:
            config_data = yaml.safe_load(f) or {}

    if vault:
        config_data["vault"] = str(vault.expanduser().resolve())

    if embedder:
        if embedder not in ("ollama", "gemini"):
            raise typer.BadParameter(f"Unknown embedder: {embedder}. Use 'ollama' or 'gemini'.")
        config_data.setdefault("embedder", {})["type"] = embedder

    if model:
        config_data.setdefault("embedder", {})["model"] = model

    if not config_data or (not vault and not embedder and not model):
        # Interactive mode - ask for settings
        typer.echo("Configure obsidian-semantic\n")

        # Vault path
        default_vault = config_data.get("vault", "")
        vault_input = typer.prompt("Vault path", default=default_vault or "~/Documents/Obsidian")
        config_data["vault"] = str(Path(vault_input).expanduser().resolve())

        # Embedder type
        current_embedder = config_data.get("embedder", {}).get("type", "ollama")
        typer.echo("\nEmbedder options: ollama, gemini")
        embedder_input = typer.prompt("Embedder type", default=current_embedder)
        if embedder_input not in ("ollama", "gemini"):
            raise typer.BadParameter(f"Unknown embedder: {embedder_input}")
        config_data.setdefault("embedder", {})["type"] = embedder_input

        if embedder_input == "ollama":
            current_model = config_data.get("embedder", {}).get("model", "nomic-embed-text")
            model_input = typer.prompt("Model name", default=current_model)
            config_data["embedder"]["model"] = model_input
            current_endpoint = config_data.get("embedder", {}).get("endpoint", "http://localhost:11434")
            endpoint = typer.prompt("Ollama endpoint", default=current_endpoint)
            config_data["embedder"]["endpoint"] = endpoint
        elif embedder_input == "gemini":
            current_model = config_data.get("embedder", {}).get("model", "text-embedding-004")
            model_input = typer.prompt("Model name", default=current_model)
            config_data["embedder"]["model"] = model_input
            typer.echo("Note: Set GEMINI_API_KEY environment variable for authentication.")

    # Write config
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    typer.echo(f"Configuration saved to {config_file}")


if __name__ == "__main__":
    app()
