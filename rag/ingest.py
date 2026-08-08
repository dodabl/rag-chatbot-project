from pypdf import PdfReader
import numpy as np
import faiss
from embedder import Embedder
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

class PDFIngestor:
    def __init__(self, pdf_path: str, embedder: Embedder | None = None):
        self.pdf_path = pdf_path
        self.embedder = embedder or Embedder()
    def extract_chunks(self) -> list[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        reader = PdfReader(self.pdf_path)
        full_text: str = ""
        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue
            full_text += text
        chunks: list[str] = text_splitter.split_text(full_text)
        return chunks
    
    def embed_chunks(self, chunks: list[str] | str) -> np.ndarray:
        embeddings: np.ndarray = self.embedder.encode(chunks)
        return embeddings

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def ingest(self) -> tuple[list[str], np.ndarray, faiss.IndexFlatL2]:
        chunks: list[str] = self.extract_chunks()
        embeddings: np.ndarray = self.embed_chunks(chunks)
        index: faiss.IndexFlatL2 = self.create_faiss_index(embeddings)

        faiss.write_index(index, "../data/vector_store/index.faiss")
        with open("../data/vector_store/chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return chunks, embeddings, index