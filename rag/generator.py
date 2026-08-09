from .retriever import Retriever
import requests
from config import OLLAMA_API_GENERATE_URL
from config import VECTOR_STORE_DIR
from pathlib import Path
import asyncio
class Generator:
    def __init__(self, chunks_path: Path = VECTOR_STORE_DIR / "chunks.json"):
        self.chunks_path = chunks_path
    async def get_retrieved_chunks(self, query: str) -> list[str] | str:
        _retriever = Retriever(chunks_path=self.chunks_path)
        retrieved_chunks = await _retriever.retrieve(query=query)
        return retrieved_chunks
    async def get_response(self, query: str) -> tuple[requests.Response, list[str] | str]:
        retrieved_chunks = await self.get_retrieved_chunks(query=query)
        request_body = {
            "model": "llama3",
            "prompt": f"Generate response using only these text chunks: {retrieved_chunks}.\nprompt: {query}",
            "stream": False
        }
        response = await asyncio.to_thread(requests.post, OLLAMA_API_GENERATE_URL, json=request_body)
        return response, retrieved_chunks
    async def generate(self, query: str) -> tuple[str, list[str] | str]:
        response, chunks = await self.get_response(query=query)
        return response.json()["response"], chunks