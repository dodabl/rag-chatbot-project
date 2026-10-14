import numpy as np
from .embedder import Embedder
import faiss
import json
from config import VECTOR_STORE_DIR
import asyncio
import aiofiles
from logging_config import get_logger, Timer

logger = get_logger(__name__)
class Retriever:
    def __init__(self, doc_id: str, embedder: Embedder):
        self.doc_id = doc_id
        self.embedder = embedder

    async def load_index_chunks(self) -> tuple[faiss.Index, dict[int, str]]:
        vector_store_dir = VECTOR_STORE_DIR / self.doc_id
        index_path = vector_store_dir / "index.faiss"
        chunks_path = vector_store_dir / "chunks.json"
        try:
            async with Timer() as t:
                index = await asyncio.to_thread(faiss.read_index, str(index_path))
            logger.info(
                "faiss index loaded to memory",
                extra={
                    "doc_id": self.doc_id,
                    "duration_ms": t.elapsed_ms
                }
            )
            async with Timer() as t:
                async with aiofiles.open(chunks_path, "r", encoding="utf-8") as f:
                    content: str = await f.read()
                chunks = await asyncio.to_thread(json.loads, content)
            logger.info(
                "chunks loaded",
                extra={
                    "doc_id": self.doc_id,
                    "duration_ms": t.elapsed_ms
                }
            )
            return index, chunks
        except Exception as e:
            logger.error(
                "Failed to load FAISS index or chunks",
                extra={
                    "doc_id": self.doc_id,
                    "path": vector_store_dir
                }
            )
            raise e
    async def embed_query(self, query: str) -> np.ndarray:
        try:
            query_embedding = await self.embedder.encode(query)
            logger.info(
                "query embedded succesfully"
            )
            return query_embedding
        except ValueError:
            logger.error(
                "empty query encountered"
            )
            raise ValueError("empty query")
    async def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        index, chunks = await self.load_index_chunks()
        query_embedding = await self.embed_query(query)
        _, indices = await asyncio.to_thread(index.search, query_embedding, top_k)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        return retrieved_chunks