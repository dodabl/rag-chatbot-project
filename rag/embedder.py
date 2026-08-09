import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio

class Embedder:
    _model: SentenceTransformer | None = None
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    async def load(self) -> SentenceTransformer:
        if Embedder._model is None:
            Embedder._model = await asyncio.to_thread(SentenceTransformer, self.model_name)
        return Embedder._model

    async def encode(self, texts: list[str] | str) -> np.ndarray:
        model = await self.load()
        embeddings: np.ndarray = await asyncio.to_thread(
            model.encode,
            texts,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True    
        )
        if isinstance(texts, str):
            return np.array(embeddings, dtype=np.float32).reshape(1, -1)
        return np.array(embeddings, dtype=np.float32)