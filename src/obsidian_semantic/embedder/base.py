"""Abstract base class for embedders."""

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Protocol for embedding text into vectors.

    All embedder implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension.

        Returns:
            Number of dimensions in the embedding vector.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier.

        Returns:
            String identifying the embedding model.
        """
        ...
