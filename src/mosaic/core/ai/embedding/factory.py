from enum import Enum
from typing import Optional

from .base_embedding import BaseEmbeddingModel
from .openai_embedding import OpenAIEmbeddingModel


class EmbeddingProvider(Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


# Map providers to their default models
DEFAULT_MODELS = {
    EmbeddingProvider.OPENAI: "text-embedding-3-small",
    EmbeddingProvider.SENTENCE_TRANSFORMERS: "all-MiniLM-L6-v2",
}


def create_embedding_model(
    provider: EmbeddingProvider,
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
) -> BaseEmbeddingModel:
    """
    Factory function to create embedding models.

    Args:
        provider: The provider to use
        model_name: The model name to use (optional, uses default if not provided)
        dimension: The embedding dimension (optional)

    Returns:
        An embedding model instance

    Raises:
        ImportError: If required dependencies are not installed
    """
    if provider in [EmbeddingProvider.OPENAI]:
        return OpenAIEmbeddingModel(model_name=model_name or DEFAULT_MODELS[provider], dimension=dimension)

    elif provider in [EmbeddingProvider.SENTENCE_TRANSFORMERS]:
        try:
            from .sentence_transformer_embedding import SentenceTransformerEmbeddingModel

            return SentenceTransformerEmbeddingModel(model_name=model_name or DEFAULT_MODELS[provider], dimension=dimension)
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Install it with: pip install mosaic[sentence-transformers]")

    else:
        # This should never happen with proper typing, but kept for safety
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {', '.join(p.value for p in DEFAULT_MODELS.keys())}")


# Helper function to get available providers
def get_available_providers() -> list[EmbeddingProvider]:
    """
    Get a list of all available embedding providers.

    Returns:
        List of provider names that can be used with create_embedding_model
    """
    providers: list[EmbeddingProvider] = [EmbeddingProvider.OPENAI]

    try:
        import sentence_transformers  # type: ignore

        providers.extend([EmbeddingProvider.SENTENCE_TRANSFORMERS])
    except ImportError:
        pass

    return providers
