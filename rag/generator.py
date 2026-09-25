from .retriever import Retriever
from config import OLLAMA_HOST, OLLAMA_API_GENERATE_URL, MODEL
import asyncio
import httpx
from typing import AsyncGenerator
import json
from logging_config import get_logger, Timer
logger = get_logger(__name__)


class Generator:
    def __init__(self): 
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100)
        )
        logger.info(
            "HTTP client initialized",
            extra={
                "ollama_host": OLLAMA_HOST,
                "ollama_api_generate_url": OLLAMA_API_GENERATE_URL,
                "model": MODEL
            }
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
            "keep_alive": -1
        }
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
        response = await self.client.get(f"{OLLAMA_HOST}/api/version")
        if response.status_code != 200:
            logger.critical(
                "Ollama API check failed",
                extra={
                    "ollama_host": OLLAMA_HOST,
                    "status_code": response.status_code,
                    "response_text": response.text
                }
            )
            return False
        logger.debug(
            "Ollama API running",
            extra={
                "ollama_host": OLLAMA_HOST,
                "ollama_version": response.json().get("version")
            }
        )
        return True

    async def close(self):
            await self.client.aclose()