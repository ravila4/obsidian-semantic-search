"""Embedder implementations for semantic search."""

import os

from obsidian_semantic.embedder.base import Embedder
from obsidian_semantic.embedder.ollama import OllamaEmbedder

__all__ = ["Embedder", "OllamaEmbedder", "create_embedder"]


def create_embedder(
    embedder_type: str | None = None,
    **kwargs,
) -> Embedder:
    """Create an embedder instance based on type.

    Args:
        embedder_type: Type of embedder ("ollama" or "gemini").
            If not provided, reads from OBSIDIAN_EMBEDDER env var,
            defaulting to "ollama".
        **kwargs: Additional arguments passed to the embedder constructor.

    Returns:
        Configured Embedder instance.

    Raises:
        ValueError: If embedder_type is unknown.
    """
    if embedder_type is None:
        embedder_type = os.environ.get("OBSIDIAN_EMBEDDER", "ollama")

    embedder_type = embedder_type.lower()

    if embedder_type == "ollama":
        return OllamaEmbedder(**kwargs)
    elif embedder_type == "gemini":
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        return GeminiEmbedder(**kwargs)
    else:
        raise ValueError(
            f"Unknown embedder type: {embedder_type}. "
            "Supported types: ollama, gemini"
        )
