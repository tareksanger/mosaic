import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union

from .base_embedding import BaseEmbeddingModel, Embedding

import numpy as np


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: Optional[int] = None):
        super().__init__(model_name)

        # Import check with better error message
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError:
            raise ImportError("sentence-transformers is not installed. Install it with: pip install mosaic[sentence-transformers]")

        self.model = SentenceTransformer(model_name)
        self._dimension = dimension or self.model.get_sentence_embedding_dimension()
        self.executor = ThreadPoolExecutor()

    def embed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]
        embeddings = self.model.encode(text, convert_to_numpy=True)
        return np.array(embeddings)

    async def aembed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(self.executor, self.model.encode, text)
        return np.array(embeddings)
