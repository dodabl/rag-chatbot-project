from pypdf import PdfReader
import numpy as np
import faiss
from .embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import aiofiles
import asyncio
from pathlib import Path
from config import UPLOADS_DIR, VECTOR_STORE_DIR

class PDFIngestor:
    def __init__(self, doc_id: str, embedder: Embedder | None = None):
        self.doc_id = doc_id
        self.embedder = embedder or Embedder()
    async def extract_chunks(self) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        path = UPLOADS_DIR / f"{self.doc_id}.pdf"
        reader = await asyncio.to_thread(PdfReader, path)
        full_text: str = ""
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            full_text += text
        chunks: list[str] = await asyncio.to_thread(text_splitter.split_text, full_text)
        return chunks
    
    async def embed_chunks(self, chunks: list[str] | str) -> np.ndarray:
        embeddings: np.ndarray = await self.embedder.encode(chunks)
        return embeddings

    async def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        index = await asyncio.to_thread(faiss.IndexFlatL2, embeddings.shape[1])
        await asyncio.to_thread(index.add, embeddings)
        return index

    async def ingest(self) -> tuple[Path, Path]:
        chunks: list[str] = await self.extract_chunks()
        embeddings: np.ndarray = await self.embed_chunks(chunks)
        index: faiss.IndexFlatL2 = await self.create_faiss_index(embeddings)
        index_path = VECTOR_STORE_DIR / self.doc_id / "index.faiss"
        chunks_path = VECTOR_STORE_DIR / self.doc_id / "chunks.json"
        await asyncio.to_thread(faiss.write_index, index, str(index_path))
        async with aiofiles.open(chunks_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(chunks, ensure_ascii=False, indent=2))
        return chunks_path, index_path