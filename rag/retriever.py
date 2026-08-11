import numpy as np
from .embedder import Embedder
import faiss
import json
from config import VECTOR_STORE_DIR
from pathlib import Path
import asyncio
import aiofiles
class Retriever:
    def __init__(self, chunks_path: Path, index_path: Path = VECTOR_STORE_DIR / "index.faiss"):
        self.index_path = index_path
        self.chunks_path = chunks_path

    async def load_index_chunks(self) -> tuple[faiss.Index, dict[int, str]]:
        index = await asyncio.to_thread(faiss.read_index, str(self.index_path))
        async with aiofiles.open(self.chunks_path, "r", encoding="utf-8") as f:
            content: str = await f.read()
        chunks = await asyncio.to_thread(json.loads, content)
        return index, chunks
    async def embed_query(self, query: str) -> np.ndarray:
        _embedder = Embedder()
        query_embedding = await _embedder.encode(query)
        return query_embedding
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        index, chunks = await self.load_index_chunks()
        query_embedding = await self.embed_query(query)
        _, indices = await asyncio.to_thread(index.search, query_embedding, top_k)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        return retrieved_chunks