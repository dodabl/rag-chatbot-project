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
        index_path = VECTOR_STORE_DIR / f"{self.doc_id}/index.faiss"
        chunks_path = VECTOR_STORE_DIR / f"{self.doc_id}/chunks.json"
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
        except Exception:
            logger.error(
                "could not read FAISS index from disk",
                extra={
                    "doc_id": self.doc_id
                }
            )
        try:
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
        except Exception as e:
            logger.error(
                "Failed to load chunks",
                extra={
                    "doc_id": self.doc_id,
                    "path": chunks_path
                }
            )
            raise e
        return index, chunks
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