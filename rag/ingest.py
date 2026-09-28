from pypdf import PdfReader
import numpy as np
import faiss
from .embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import aiofiles
import asyncio
from config import UPLOADS_DIR, VECTOR_STORE_DIR
from logging_config import get_logger, Timer
import os

logger = get_logger(__name__)
EMBEDDER = Embedder()
class PDFIngestor:
    def __init__(self, doc_id: str | None = None):
        self.doc_id = doc_id


    def set_doc_id(self, doc_id: str):
        self.doc_id = doc_id


    async def extract_chunks(self) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        path = UPLOADS_DIR / f"{self.doc_id}.pdf"
        try:
            reader = await asyncio.to_thread(PdfReader, path)
        except Exception as e:
            logger.error(
                "Failed to read PDF file",
                extra={
                    "doc_id": self.doc_id,
                    "path": path
                }
            )
            raise e
        full_text: str = ""
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            full_text += text
        chunks: list[str] = await asyncio.to_thread(text_splitter.split_text, full_text)
        return chunks

    
    async def embed_chunks(self, chunks: list[str] | str) -> np.ndarray:
        try:
            embeddings: np.ndarray = await EMBEDDER.encode(chunks)
            return embeddings
        except ValueError as e:
            logger.error(
                "empty string passed to embed_chunks()"
            )
            raise e

    async def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        index = await asyncio.to_thread(faiss.IndexFlatL2, embeddings.shape[1])
        await asyncio.to_thread(index.add, embeddings)
        return index

    async def ingest(self):
        async with Timer() as timer:
            chunks: list[str] = await self.extract_chunks()
            logger.info(
                "chunks extracted",
                extra={
                    "num_chunks": len(chunks)
                }
            )
            embeddings: np.ndarray = await self.embed_chunks(chunks)
            logger.info(
                "chunks embedded",
                extra={
                    "embedding_dim": embeddings.shape[1]
                }
            )
            index: faiss.IndexFlatL2 = await self.create_faiss_index(embeddings)
            logger.info(
                "faiss index created",
                extra={
                    "num_vectors": index.ntotal
                }
            )
            index_path = VECTOR_STORE_DIR / str(self.doc_id) / "index.faiss"
            chunks_path = VECTOR_STORE_DIR / str(self.doc_id) / "chunks.json"
            try:
                await asyncio.to_thread(faiss.write_index, index, str(index_path))
                logger.info(
                    "faiss index written to disk",
                    extra={
                        "index_path": index_path
                    }
                )
            except Exception:
                logger.error(
                    "could not write FAISS index to disk",
                    extra={
                        "doc_id": self.doc_id
                    }
                )
                raise
            try:
                async with aiofiles.open(chunks_path, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(chunks, ensure_ascii=False, indent=2))
                logger.info(
                    "chunks written to disk",
                    extra={
                        "chunks_path": chunks_path
                    }
                )
            except Exception as e:
                os.remove(index_path)
                if chunks_path.is_file():
                    os.remove(chunks_path)
                logger.error(
                    "could not write chunks to disk",
                    extra={
                        "doc_id": self.doc_id
                    }
                )
                raise e
        logger.info(
            "Ingestion complete",
            extra={
                "duration_ms": timer.elapsed_ms,
                "doc_id": self.doc_id
            }
        )