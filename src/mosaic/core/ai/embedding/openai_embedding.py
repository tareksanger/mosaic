from typing import Optional, Union

from .base_embedding import BaseEmbeddingModel, Embedding

import numpy as np
from openai import AsyncOpenAI, OpenAI


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small", dimension: Optional[int] = None):
        super().__init__(model_name)
        self.client = OpenAI()
        self.aclient = AsyncOpenAI()
        # Set the dimension based on the model; for example:
        self._dimension = dimension or 1536

    def embed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]

        if len(text) == 0:
            return np.array([])

        response = self.client.embeddings.create(model=self.model_name, input=text)
        embeddings = [np.array([e.embedding]) for e in response.data]
        return np.vstack(embeddings)

    async def aembed(self, text: Union[str, list[str]]) -> Embedding:
        if isinstance(text, str):
            text = [text]

        if len(text) == 0:
            return np.array([])

        response = await self.aclient.embeddings.create(model=self.model_name, input=text)
        embeddings = [np.array([e.embedding]) for e in response.data]
        return np.vstack(embeddings)
