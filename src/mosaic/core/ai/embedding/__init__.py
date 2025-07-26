from .base_embedding import BaseEmbeddingModel, Embedding, SimilarityMode
from .factory import create_embedding_model, EmbeddingProvider, get_available_providers
from .openai_embedding import OpenAIEmbeddingModel

__all__ = [
    "BaseEmbeddingModel",
    "Embedding",
    "SimilarityMode",
    "create_embedding_model",
    "EmbeddingProvider",
    "get_available_providers",
    "OpenAIEmbeddingModel",
]

try:
    # Import sentence_transformers to check if it's installed before adding SentenceTransformerEmbeddingModel to __all__
    import sentence_transformers  # type: ignore  # noqa: I001

    # Import sentence_transformers to check if it's installed before adding SentenceTransformerEmbeddingModel to __all__
    from .sentence_transformer_embedding import SentenceTransformerEmbeddingModel

    __all__.append("SentenceTransformerEmbeddingModel")

except ImportError:
    pass
