"""Tests for the embedder protocol and implementations."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock

from obsidian_semantic.embedder.base import Embedder
from obsidian_semantic.embedder.ollama import OllamaEmbedder


class TestEmbedderProtocol:
    """Test that Embedder is a proper protocol/ABC."""

    def test_embedder_is_abstract(self):
        """Cannot instantiate Embedder directly."""
        with pytest.raises(TypeError):
            Embedder()  # type: ignore

    def test_embedder_requires_embed_method(self):
        """Subclasses must implement embed()."""

        class BadEmbedder(Embedder):
            @property
            def dimension(self) -> int:
                return 768

            @property
            def model_name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            BadEmbedder()

    def test_embedder_requires_dimension(self):
        """Subclasses must implement dimension property."""

        class BadEmbedder(Embedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            @property
            def model_name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            BadEmbedder()

    def test_embedder_requires_model_name(self):
        """Subclasses must implement model_name property."""

        class BadEmbedder(Embedder):
            def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            @property
            def dimension(self) -> int:
                return 768

        with pytest.raises(TypeError):
            BadEmbedder()


class TestOllamaEmbedder:
    """Test the Ollama embedder implementation."""

    def test_default_configuration(self):
        """Should have sensible defaults."""
        embedder = OllamaEmbedder()

        assert embedder.model_name == "nomic-embed-text"
        assert embedder.dimension == 768
        assert embedder._endpoint == "http://localhost:11434"
        assert embedder._batch_size == 32

    def test_custom_configuration(self):
        """Should accept custom config."""
        embedder = OllamaEmbedder(
            model="mxbai-embed-large",
            endpoint="http://192.168.1.100:11434",
            batch_size=16,
            dimension=1024,
        )

        assert embedder.model_name == "mxbai-embed-large"
        assert embedder.dimension == 1024
        assert embedder._endpoint == "http://192.168.1.100:11434"
        assert embedder._batch_size == 16

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_embed_single_text(self, mock_client_cls: Mock):
        """Should embed a single text using /api/embed endpoint."""
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = Mock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()
        result = embedder.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768
        mock_client.post.assert_called_once()
        # Verify correct endpoint and request format
        call_args = mock_client.post.call_args
        assert "/api/embed" in call_args[0][0]
        assert call_args[1]["json"]["input"] == ["Hello world"]

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_embed_multiple_texts_in_single_batch(self, mock_client_cls: Mock):
        """Should embed multiple texts in one API call when under batch_size."""
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768] * 3}
        mock_response.raise_for_status = Mock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder(batch_size=10)
        result = embedder.embed(["Text 1", "Text 2", "Text 3"])

        assert len(result) == 3
        mock_client.post.assert_called_once()  # All in one batch

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_embed_respects_batch_size(self, mock_client_cls: Mock):
        """Should split texts into batches based on batch_size."""

        def mock_batch_response(*args, **kwargs):
            batch_size = len(kwargs["json"]["input"])
            mock_resp = Mock()
            mock_resp.json.return_value = {"embeddings": [[0.1] * 768] * batch_size}
            mock_resp.raise_for_status = Mock()
            return mock_resp

        mock_client = MagicMock()
        mock_client.post.side_effect = mock_batch_response
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder(batch_size=2)
        texts = ["Text " + str(i) for i in range(5)]
        result = embedder.embed(texts)

        assert len(result) == 5
        # 5 texts with batch_size=2 means 3 API calls (2+2+1)
        assert mock_client.post.call_count == 3

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_embed_empty_list(self, mock_client_cls: Mock):
        """Should handle empty input."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()
        result = embedder.embed([])

        assert result == []
        mock_client.post.assert_not_called()

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_connection_error_raised(self, mock_client_cls: Mock):
        """Should raise ConnectionError when Ollama is unreachable."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()
        with pytest.raises(ConnectionError) as exc_info:
            embedder.embed(["Hello"])
        assert "Failed to connect to Ollama" in str(exc_info.value)

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_api_error_raised(self, mock_client_cls: Mock):
        """Should raise on API error."""
        import httpx

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Bad Request",
            request=Mock(),
            response=Mock(status_code=400),
        )

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()
        with pytest.raises(RuntimeError):
            embedder.embed(["Hello"])

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_timeout_error_raised(self, mock_client_cls: Mock):
        """Should raise TimeoutError when request times out."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()
        with pytest.raises(TimeoutError) as exc_info:
            embedder.embed(["Hello"])
        assert "timed out" in str(exc_info.value)

    @patch("obsidian_semantic.embedder.ollama.httpx.Client")
    def test_connection_pooling(self, mock_client_cls: Mock):
        """Should reuse HTTP client across multiple embed calls."""
        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_response.raise_for_status = Mock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        embedder = OllamaEmbedder()

        # Make multiple calls
        embedder.embed(["Text 1"])
        embedder.embed(["Text 2"])
        embedder.embed(["Text 3"])

        # Client should be created once
        mock_client_cls.assert_called_once()
        # Should use the same client for all calls
        assert mock_client.post.call_count == 3


class TestGeminiEmbedder:
    """Test the Gemini embedder implementation."""

    def test_default_configuration(self):
        """Should have sensible defaults."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(api_key="test-key")

        assert embedder.model_name == "models/gemini-embedding-001"
        assert embedder.dimension == 3072

    def test_custom_configuration(self):
        """Should accept custom config."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(
            api_key="test-key",
            model="models/embedding-001",
            dimension=768,
            batch_size=50,
        )

        assert embedder.model_name == "models/embedding-001"
        assert embedder.dimension == 768

    @patch("obsidian_semantic.embedder.gemini.httpx.post")
    def test_embed_single_text(self, mock_post: Mock):
        """Should embed a single text using REST API."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        mock_response = Mock()
        mock_response.json.return_value = {"embeddings": [{"values": [0.1] * 768}]}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        embedder = GeminiEmbedder(api_key="test-key")
        result = embedder.embed(["Hello world"])

        assert len(result) == 1
        assert len(result[0]) == 768
        mock_post.assert_called_once()
        # Verify API key is passed
        assert mock_post.call_args[1]["params"]["key"] == "test-key"

    @patch("obsidian_semantic.embedder.gemini.httpx.post")
    def test_embed_multiple_texts_batched(self, mock_post: Mock):
        """Should embed multiple texts in a single batch API call."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        mock_response = Mock()
        mock_response.json.return_value = {
            "embeddings": [{"values": [0.1] * 768} for _ in range(3)]
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        embedder = GeminiEmbedder(api_key="test-key")
        result = embedder.embed(["Text 1", "Text 2", "Text 3"])

        assert len(result) == 3
        # All texts should be in one batch
        mock_post.assert_called_once()
        requests = mock_post.call_args[1]["json"]["requests"]
        assert len(requests) == 3

    @patch("obsidian_semantic.embedder.gemini.httpx.post")
    def test_embed_empty_list(self, mock_post: Mock):
        """Should handle empty input."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        embedder = GeminiEmbedder(api_key="test-key")
        result = embedder.embed([])

        assert result == []
        mock_post.assert_not_called()

    def test_reads_api_key_from_env(self):
        """Should read API key from GEMINI_API_KEY or GOOGLE_API_KEY."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}, clear=False):
            embedder = GeminiEmbedder()
            assert embedder._api_key == "env-key"

    def test_raises_without_api_key(self):
        """Should raise if no API key provided or in env."""
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        with patch.dict(os.environ, {}, clear=True):
            # Clear both possible env vars
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("GOOGLE_API_KEY", None)
            with pytest.raises(ValueError, match="API key"):
                GeminiEmbedder()


class TestEmbedderFactory:
    """Test the embedder factory and configuration."""

    def test_create_ollama_embedder(self):
        """Should create Ollama embedder."""
        from obsidian_semantic.embedder import create_embedder

        embedder = create_embedder("ollama")

        assert isinstance(embedder, OllamaEmbedder)

    def test_create_gemini_embedder(self):
        """Should create Gemini embedder with API key."""
        from obsidian_semantic.embedder import create_embedder
        from obsidian_semantic.embedder.gemini import GeminiEmbedder

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            embedder = create_embedder("gemini")

        assert isinstance(embedder, GeminiEmbedder)

    def test_create_from_env_var(self):
        """Should read embedder type from OBSIDIAN_EMBEDDER env var."""
        from obsidian_semantic.embedder import create_embedder

        with patch.dict(os.environ, {"OBSIDIAN_EMBEDDER": "ollama"}):
            embedder = create_embedder()

        assert isinstance(embedder, OllamaEmbedder)

    def test_default_to_ollama(self):
        """Should default to Ollama if no env var set."""
        from obsidian_semantic.embedder import create_embedder

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OBSIDIAN_EMBEDDER", None)
            embedder = create_embedder()

        assert isinstance(embedder, OllamaEmbedder)

    def test_invalid_embedder_type(self):
        """Should raise on unknown embedder type."""
        from obsidian_semantic.embedder import create_embedder

        with pytest.raises(ValueError, match="Unknown embedder"):
            create_embedder("unknown")
