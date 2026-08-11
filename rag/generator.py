from .retriever import Retriever
from config import OLLAMA_API_GENERATE_URL
from config import VECTOR_STORE_DIR
from pathlib import Path
import httpx
from typing import AsyncGenerator
import json
class Generator:
    def __init__(self, chunks_path: Path = VECTOR_STORE_DIR / "chunks.json"):
        self.chunks_path = chunks_path
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=0)
        )


    async def get_retrieved_chunks(self, query: str) -> list[str] | str:
        _retriever = Retriever(chunks_path=self.chunks_path)
        retrieved_chunks = await _retriever.retrieve(query=query)
        return retrieved_chunks
    
    async def generate_stream(self, query: str) -> AsyncGenerator[str, None]:
        retrieved_chunks = await self.get_retrieved_chunks(query=query)
        request_body = {
            "model": "llama3",
            "prompt": f"Generate response using only these text chunks: {retrieved_chunks}.\n Query: {query}",
            "stream": True,
            "keep_alive": "10m"
        }
        async with self.client.stream("POST", OLLAMA_API_GENERATE_URL, json=request_body) as response:
            for attempt in range(2):
                yielded_anything = False
                try: 
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        if chunk.get("response"):
                            yielded_anything = True
                            yield chunk["response"]
                        if chunk.get("done"):
                            break
                    return
                except httpx.RemoteProtocolError as e:
                    if attempt == 0 and not yielded_anything:
                        continue
                    raise
    async def close(self):
            await self.client.aclose()