"""LM Studio embedder implementation (OpenAI-compatible API)."""

import httpx

from obsidian_semantic.embedder.base import Embedder


class LMStudioEmbedder(Embedder):
    """Embedder using LM Studio's OpenAI-compatible local API.

    LM Studio exposes embedding models at /v1/embeddings on port 1234
    by default. Default model is text-embedding-nomic-embed-text-v1.5
    (768 dimensions).

    Start the server with: ``lms server start``
    """

    def __init__(
        self,
        model: str = "text-embedding-nomic-embed-text-v1.5",
        endpoint: str = "http://localhost:1234",
        batch_size: int = 32,
        dimension: int = 768,
        timeout: float = 30.0,
        query_prefix: str = "",
        document_prefix: str = "",
    ):
        """Initialize the LM Studio embedder.

        Args:
            model: LM Studio model identifier for embeddings.
            endpoint: LM Studio server base URL (without /v1).
            batch_size: Number of texts to send per request.
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
        """Generate embeddings for texts via LM Studio's OpenAI-compatible API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors, one per input in the original order.

        Raises:
            ConnectionError: If unable to connect to LM Studio.
            TimeoutError: If a request exceeds the configured timeout.
            RuntimeError: If the API returns an error or unexpected response.
        """
        if not texts:
            return []

        embeddings: list[list[float]] = []
        url = f"{self._endpoint}/v1/embeddings"

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            try:
                response = self._client.post(
                    url,
                    json={"model": self._model, "input": batch},
                )
                response.raise_for_status()
                data = response.json()
                if "data" not in data:
                    raise RuntimeError(
                        f"LM Studio response missing 'data': {data}"
                    )
                # OpenAI spec: data items carry an `index` for ordering. LM
                # Studio currently returns them in request order, but sort
                # defensively so we never desync vectors from inputs.
                items = sorted(
                    data["data"], key=lambda item: item.get("index", 0)
                )
                embeddings.extend(item["embedding"] for item in items)
            except httpx.ConnectError as e:
                raise ConnectionError(
                    f"Failed to connect to LM Studio at {self._endpoint}. "
                    "Is the server running? Try: `lms server start`"
                ) from e
            except httpx.TimeoutException as e:
                raise TimeoutError(f"LM Studio request timed out: {e}") from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"LM Studio API error: {e}") from e

        return embeddings

    def embed_document(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as documents, prepending document_prefix if configured."""
        if self._document_prefix:
            return self.embed([self._document_prefix + t for t in texts])
        return self.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as queries, prepending query_prefix if configured."""
        if self._query_prefix:
            return self.embed([self._query_prefix + t for t in texts])
        return self.embed(texts)
