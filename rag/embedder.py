import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEFAULT_SENTENCE_TRANSFORMER_MODEL
from logging_config import get_logger
import asyncio
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)
    
class Embedder:
    def __init__(self, model_name: str | None = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
        self.model_name = model_name
        try:
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}': {e}. Falling back to default model '{DEFAULT_SENTENCE_TRANSFORMER_MODEL}'."
            )
            self._model = SentenceTransformer(DEFAULT_SENTENCE_TRANSFORMER_MODEL)
        finally:
            logger.info(
                "Model loaded successfully."
            )
    async def encode(self, texts: list[str] | str) -> np.ndarray:
        if texts == "":
            raise ValueError("empty string")
        embeddings: np.ndarray = await asyncio.to_thread(
            self._model.encode,
            texts,
            convert_to_numpy=True,
            convert_to_tensor=False,
            normalize_embeddings=True    
        )
        logger.info(
            f"Encoded {len(texts) if isinstance(texts, list) else 1} text(s) into embeddings of shape {embeddings.shape}."
        )
        if isinstance(texts, str):
            return np.array(embeddings, dtype=np.float32).reshape(1, -1)
        return np.array(embeddings, dtype=np.float32)