from .retriever import Retriever
from config import OLLAMA_HOST, OLLAMA_API_GENERATE_URL, MODEL
import httpx
from typing import AsyncGenerator
import json
from logging_config import get_logger, Timer
logger = get_logger(__name__)

class Generator:
    def __init__(self): 
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(max_keepalive_connections=0)
        )


    async def get_retrieved_chunks(self, doc_id: str, query: str) -> list[str] | str:
        _retriever = Retriever(doc_id)
        retrieved_chunks = await _retriever.retrieve(query=query)
        return retrieved_chunks
    
    async def generate_stream(self, doc_id: str, query: str) -> AsyncGenerator[str, None]:
        retrieved_chunks = await self.get_retrieved_chunks(doc_id=doc_id, query=query)
        request_body = {
            "model": MODEL,
            "prompt": f"Generate response using only these text chunks: {retrieved_chunks}.\n Query: {query}",
            "stream": True,
            "keep_alive": "10m"
        }
        n_tokens = 0
        async with Timer() as timer:
            async with self.client.stream("POST", OLLAMA_API_GENERATE_URL, json=request_body) as response:

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if chunk.get("response"):
                        yield chunk["response"]
                    if chunk.get("done"):
                        n_tokens = chunk.get("eval_count", 0)
                        break
        logger.info(
            "Generation complete",
            extra={
                "duration_ms": timer.elapsed_ms,
                "n_tokens": n_tokens,
                "model": MODEL,
            }
        )
    async def ollama_check(self) -> bool:
        try:
            response = await self.client.get(f"{OLLAMA_HOST}/api/version")
            logger.debug(
                "Ollama API running",
                extra={
                    "ollama_host": OLLAMA_HOST,
                    "ollama_version": response.json().get("version")
                }
            )
            return response.status_code == 200
        except Exception:
            logger.error(
                "Failed to connect to Ollama API",
                extra={
                    "ollama_host": OLLAMA_HOST
                }
            )
            return False

    async def close(self):
            await self.client.aclose()