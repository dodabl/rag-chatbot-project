import numpy as np
from .embedder import Embedder
import faiss
import json
from config import VECTOR_STORE_DIR
import asyncio
import aiofiles
class Retriever:
    def __init__(self, doc_id: str):
        self.doc_id = doc_id

    async def load_index_chunks(self) -> tuple[faiss.Index, dict[int, str]]:
        index_path = VECTOR_STORE_DIR / f"{self.doc_id}/index.faiss"
        chunks_path = VECTOR_STORE_DIR / f"{self.doc_id}/chunks.json"
        index = await asyncio.to_thread(faiss.read_index, str(index_path))
        async with aiofiles.open(chunks_path, "r", encoding="utf-8") as f:
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