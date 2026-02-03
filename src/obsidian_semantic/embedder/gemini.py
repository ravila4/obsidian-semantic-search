"""Gemini embedder implementation using REST API."""

import os

import httpx

from obsidian_semantic.embedder.base import Embedder


class GeminiEmbedder(Embedder):
    """Embedder using Google's Gemini API via REST.

    Default model is text-embedding-004 (768 dimensions).
    Requires GEMINI_API_KEY or GOOGLE_API_KEY environment variable,
    or api_key passed directly.
    """

    API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "models/text-embedding-004",
        dimension: int = 768,
        batch_size: int = 100,
        task_type: str = "RETRIEVAL_DOCUMENT",
        timeout: float = 30.0,
    ):
        """Initialize the Gemini embedder.

        Args:
            api_key: Gemini API key. If not provided, reads from
                GEMINI_API_KEY or GOOGLE_API_KEY environment variable.
            model: Gemini model name for embeddings.
            dimension: Embedding vector dimension.
            batch_size: Number of texts to process per API call.
            task_type: Task type for embeddings (RETRIEVAL_DOCUMENT,
                RETRIEVAL_QUERY, SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING).
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get(
            "GOOGLE_API_KEY"
        )
        if not self._api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Normalize model name to include models/ prefix
        self._model = model if model.startswith("models/") else f"models/{model}"
        self._dimension = dimension
        self._batch_size = batch_size
        self._task_type = task_type
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Return model identifier."""
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts using Gemini REST API.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If the API returns an error.
        """
        if not texts:
            return []

        embeddings = []
        url = f"{self.API_BASE}/{self._model}:batchEmbedContents"

        # Process in batches
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            requests = [
                {
                    "model": self._model,
                    "content": {"parts": [{"text": text}]},
                    "taskType": self._task_type,
                }
                for text in batch
            ]

            try:
                response = httpx.post(
                    url,
                    params={"key": self._api_key},
                    json={"requests": requests},
                    timeout=self._timeout,
                )
                response.raise_for_status()
                data = response.json()

                if "embeddings" not in data:
                    raise RuntimeError(f"Gemini response missing 'embeddings': {data}")

                for emb in data["embeddings"]:
                    embeddings.append(emb["values"])

            except httpx.ConnectError as e:
                raise ConnectionError(f"Failed to connect to Gemini API: {e}") from e
            except httpx.TimeoutException as e:
                raise TimeoutError(f"Gemini request timed out: {e}") from e
            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"Gemini API error: {e.response.text}") from e

        return embeddings
