"""Ollama embedder implementation."""

import httpx

from obsidian_semantic.embedder.base import Embedder


class OllamaEmbedder(Embedder):
    """Embedder using Ollama's local API.

    Default model is nomic-embed-text (768 dimensions).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        endpoint: str = "http://localhost:11434",
        batch_size: int = 32,
        dimension: int = 768,
        timeout: float = 30.0,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        """Initialize the Ollama embedder.

        Args:
            model: Ollama model name for embeddings.
            endpoint: Ollama API endpoint URL.
            batch_size: Number of texts to process per batch (for rate limiting).
            dimension: Embedding vector dimension.
            timeout: Request timeout in seconds.
            query_prefix: Prefix prepended to texts in embed_query().
            document_prefix: Prefix prepended to texts in embed_document().
        """
        self._model = model
        self._endpoint = endpoint
        self._batch_size = batch_size
        self._dimension = dimension
        self._timeout = timeout
        self._query_prefix = query_prefix
        self._document_prefix = document_prefix
        self._client = httpx.Client(timeout=timeout)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model

    def close(self) -> None:
        """Close the HTTP client and release resources."""
        self._client.close()

    def __del__(self) -> None:
        """Cleanup HTTP client on garbage collection."""
        if hasattr(self, "_client"):
            self._client.close()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Ollama API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            ConnectionError: If unable to connect to Ollama.
            RuntimeError: If the API returns an error or unexpected response.
        """
        if not texts:
            return []

        embeddings = []
        url = f"{self._endpoint}/api/embed"

        # Process texts in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.post(
                    url,
                    json={"model": self._model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()
                if "embeddings" not in data:
                    raise RuntimeError(f"Ollama response missing 'embeddings': {data}")
                embeddings.extend(data["embeddings"])
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Failed to connect to Ollama at {self._endpoint}: {e}"
                ) from e
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Ollama request timed out: {e}") from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Ollama API error: {e}") from e

        return embeddings

    def embed_document(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as documents, prepending document_prefix if configured."""
        if self._document_prefix:
            texts = [self._document_prefix + t for t in texts]
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as queries, prepending query_prefix if configured."""
        if self._query_prefix:
            texts = [self._query_prefix + t for t in texts]
        return self.embed(texts)
