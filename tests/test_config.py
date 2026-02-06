"""Tests for configuration loading."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from obsidian_semantic.config import Config, load_config


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_embedder_type(self):
        """Should default to ollama embedder."""
        config = Config()
        assert config.embedder.type == "ollama"

    def test_default_ignore_patterns(self):
        """Should ignore Templates by default."""
        config = Config()
        assert "Templates/*" in config.ignore

    def test_default_database_path(self):
        """Should use .obsidian-semantic/index.lance by default."""
        config = Config()
        assert config.database == ".obsidian-semantic/index.lance"


class TestConfigFromYaml:
    """Test loading configuration from YAML files."""

    def test_load_from_yaml_file(self, tmp_path: Path):
        """Should load config from YAML file."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: gemini
  model: models/text-embedding-004
  batch_size: 50

ignore:
  - "Templates/*"
  - "Archive/*"
""")
        config = load_config(vault_path=tmp_path)

        assert config.embedder.type == "gemini"
        assert config.embedder.model == "models/text-embedding-004"
        assert config.embedder.batch_size == 50
        assert "Archive/*" in config.ignore

    def test_load_ollama_specific_config(self, tmp_path: Path):
        """Should load Ollama-specific settings."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: ollama
  model: nomic-embed-text
  endpoint: http://192.168.1.100:11434
""")
        config = load_config(vault_path=tmp_path)

        assert config.embedder.type == "ollama"
        assert config.embedder.endpoint == "http://192.168.1.100:11434"

    def test_missing_config_uses_defaults(self, tmp_path: Path):
        """Should use defaults when no config file exists."""
        # Patch home to avoid loading actual user config
        with patch("obsidian_semantic.config.Path.home", return_value=tmp_path):
            config = load_config(vault_path=tmp_path)

        assert config.embedder.type == "ollama"
        assert "Templates/*" in config.ignore


class TestConfigFromEnv:
    """Test environment variable overrides."""

    def test_env_overrides_embedder_type(self, tmp_path: Path):
        """OBSIDIAN_EMBEDDER should override config file."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: ollama
""")
        with patch.dict(os.environ, {"OBSIDIAN_EMBEDDER": "gemini", "GEMINI_API_KEY": "test"}):
            config = load_config(vault_path=tmp_path)

        assert config.embedder.type == "gemini"

    def test_env_provides_api_key(self, tmp_path: Path):
        """Should read API key from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "my-secret-key"}):
            config = load_config(vault_path=tmp_path)

        assert config.embedder.api_key == "my-secret-key"


class TestConfigUserDefault:
    """Test user-level default config."""

    def test_loads_user_config_when_no_vault_config(self, tmp_path: Path):
        """Should fall back to ~/.config/obsidian-semantic/config.yaml."""
        user_config_dir = tmp_path / ".config" / "obsidian-semantic"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yaml"
        user_config.write_text("""
embedder:
  type: gemini
  model: models/embedding-001
""")
        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        with patch.dict(os.environ, {"HOME": str(tmp_path), "GEMINI_API_KEY": "test"}):
            config = load_config(vault_path=vault_path)

        assert config.embedder.type == "gemini"
        assert config.embedder.model == "models/embedding-001"

    def test_vault_config_overrides_user_config(self, tmp_path: Path):
        """Vault-level config should take precedence over user config."""
        # User config
        user_config_dir = tmp_path / ".config" / "obsidian-semantic"
        user_config_dir.mkdir(parents=True)
        user_config = user_config_dir / "config.yaml"
        user_config.write_text("""
embedder:
  type: gemini
""")
        # Vault config
        vault_path = tmp_path / "vault"
        vault_path.mkdir()
        vault_config = vault_path / ".obsidian-semantic.yaml"
        vault_config.write_text("""
embedder:
  type: ollama
""")

        with patch.dict(os.environ, {"HOME": str(tmp_path)}):
            config = load_config(vault_path=vault_path)

        assert config.embedder.type == "ollama"


class TestPrefixConfig:
    """Test query_prefix and document_prefix configuration."""

    def test_prefix_fields_default_to_none(self):
        """Prefix fields should be None by default."""
        config = Config()
        assert config.embedder.query_prefix is None
        assert config.embedder.document_prefix is None

    def test_prefixes_loaded_from_yaml(self, tmp_path: Path):
        """Should load prefix strings from YAML config."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: ollama
  model: qwen3-embedding:8b
  query_prefix: "Instruct: Retrieve relevant notes\\nQuery: "
  document_prefix: ""
""")
        config = load_config(vault_path=tmp_path)
        # YAML interprets \\n as literal newline
        assert config.embedder.query_prefix == "Instruct: Retrieve relevant notes\nQuery: "
        assert config.embedder.document_prefix == ""

    def test_prefixes_passed_to_ollama_embedder(self, tmp_path: Path):
        """create_embedder() should pass prefixes to OllamaEmbedder."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: ollama
  model: nomic-embed-text
  query_prefix: "search_query: "
  document_prefix: "search_document: "
""")
        config = load_config(vault_path=tmp_path)
        embedder = config.create_embedder()

        from obsidian_semantic.embedder import OllamaEmbedder
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder._query_prefix == "search_query: "
        assert embedder._document_prefix == "search_document: "


class TestCreateEmbedderFromConfig:
    """Test creating embedder instances from config."""

    def test_create_ollama_from_config(self, tmp_path: Path):
        """Should create OllamaEmbedder from config."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: ollama
  model: mxbai-embed-large
  batch_size: 16
""")
        config = load_config(vault_path=tmp_path)
        embedder = config.create_embedder()

        from obsidian_semantic.embedder import OllamaEmbedder
        assert isinstance(embedder, OllamaEmbedder)
        assert embedder.model_name == "mxbai-embed-large"

    def test_create_gemini_from_config(self, tmp_path: Path):
        """Should create GeminiEmbedder from config."""
        config_file = tmp_path / ".obsidian-semantic.yaml"
        config_file.write_text("""
embedder:
  type: gemini
  model: models/text-embedding-004
""")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            config = load_config(vault_path=tmp_path)
            embedder = config.create_embedder()

        from obsidian_semantic.embedder.gemini import GeminiEmbedder
        assert isinstance(embedder, GeminiEmbedder)
