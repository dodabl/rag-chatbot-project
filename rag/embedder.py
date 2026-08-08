import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    _model: SentenceTransformer | None = None
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    def load(self) -> SentenceTransformer:
        if Embedder._model is None:
            Embedder._model = SentenceTransformer(self.model_name)
        return Embedder._model

    def encode(self, texts: list[str] | str) -> np.ndarray:
        model = self.load()
        embeddings: np.ndarray = model.encode(
            texts, convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True    
        )
        if isinstance(texts, str):
            return np.array(embeddings, dtype=np.float32).reshape(1, -1)
        return np.array(embeddings, dtype=np.float32)