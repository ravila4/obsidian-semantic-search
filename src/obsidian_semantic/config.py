"""Configuration management for obsidian-semantic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from obsidian_semantic.embedder.base import Embedder


@dataclass
class EmbedderConfig:
    """Embedder-specific configuration."""

    type: str = "ollama"
    model: str | None = None
    batch_size: int | None = None
    dimension: int | None = None
    timeout: float | None = None
    # Ollama-specific
    endpoint: str | None = None
    # Gemini-specific
    api_key: str | None = None
    task_type: str | None = None


@dataclass
class Config:
    """Application configuration."""

    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    database: str = ".obsidian-semantic/index.lance"
    ignore: list[str] = field(default_factory=lambda: ["Templates/*"])

    def create_embedder(self) -> Embedder:
        """Create an embedder instance from this configuration."""
        kwargs = {}

        if self.embedder.model:
            kwargs["model"] = self.embedder.model
        if self.embedder.batch_size:
            kwargs["batch_size"] = self.embedder.batch_size
        if self.embedder.dimension:
            kwargs["dimension"] = self.embedder.dimension
        if self.embedder.timeout:
            kwargs["timeout"] = self.embedder.timeout

        if self.embedder.type == "ollama":
            from obsidian_semantic.embedder import OllamaEmbedder

            if self.embedder.endpoint:
                kwargs["endpoint"] = self.embedder.endpoint
            return OllamaEmbedder(**kwargs)

        elif self.embedder.type == "gemini":
            from obsidian_semantic.embedder.gemini import GeminiEmbedder

            if self.embedder.api_key:
                kwargs["api_key"] = self.embedder.api_key
            if self.embedder.task_type:
                kwargs["task_type"] = self.embedder.task_type
            return GeminiEmbedder(**kwargs)

        else:
            raise ValueError(f"Unknown embedder type: {self.embedder.type}")


def load_config(vault_path: Path | None = None) -> Config:
    """Load configuration from files and environment.

    Config sources (in order of precedence, highest first):
    1. Environment variables
    2. Vault-level config: {vault_path}/.obsidian-semantic.yaml
    3. User-level config: ~/.config/obsidian-semantic/config.yaml
    4. Built-in defaults

    Args:
        vault_path: Path to the Obsidian vault. If provided, looks for
            vault-level config file.

    Returns:
        Merged Config object.
    """
    config = Config()

    # Load user-level config
    user_config_path = Path.home() / ".config" / "obsidian-semantic" / "config.yaml"
    if user_config_path.exists():
        config = _merge_yaml_into_config(config, user_config_path)

    # Load vault-level config (overrides user config)
    if vault_path:
        vault_config_path = Path(vault_path) / ".obsidian-semantic.yaml"
        if vault_config_path.exists():
            config = _merge_yaml_into_config(config, vault_config_path)

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    return config


def _merge_yaml_into_config(config: Config, yaml_path: Path) -> Config:
    """Merge YAML file contents into config."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return config

    # Merge embedder settings
    if "embedder" in data:
        emb = data["embedder"]
        if "type" in emb:
            config.embedder.type = emb["type"]
        if "model" in emb:
            config.embedder.model = emb["model"]
        if "batch_size" in emb:
            config.embedder.batch_size = emb["batch_size"]
        if "dimension" in emb:
            config.embedder.dimension = emb["dimension"]
        if "timeout" in emb:
            config.embedder.timeout = emb["timeout"]
        if "endpoint" in emb:
            config.embedder.endpoint = emb["endpoint"]
        if "api_key" in emb:
            config.embedder.api_key = emb["api_key"]
        if "task_type" in emb:
            config.embedder.task_type = emb["task_type"]

    # Merge other settings
    if "database" in data:
        config.database = data["database"]
    if "ignore" in data:
        config.ignore = data["ignore"]

    return config


def _apply_env_overrides(config: Config) -> Config:
    """Apply environment variable overrides to config."""
    # Embedder type
    if env_type := os.environ.get("OBSIDIAN_EMBEDDER"):
        config.embedder.type = env_type

    # API keys
    if api_key := os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        config.embedder.api_key = api_key

    return config
