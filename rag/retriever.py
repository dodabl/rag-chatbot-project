import numpy as np
from ingest import PDFIngestor
from embedder import Embedder
import faiss
import json
class Retriever:
    def __init__(self, chunks_path: str, index_path: str = "../vector_store/index.faiss"):
        self.index_path = index_path
        self.chunks_path = chunks_path

    def load_index_chunks(self) -> tuple[faiss.Index, dict[int, str]]:
        index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            f.close()
        return index, chunks
    def embed_query(self, query: str) -> np.ndarray:
        _embedder = Embedder()
        query_embedding = _embedder.encode(query)
        return query_embedding
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        index, chunks = self.load_index_chunks()
        query_embedding = self.embed_query(query)
        _, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [chunks[idx] for idx in indices[0]]
        return retrieved_chunks