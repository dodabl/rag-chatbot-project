from .retriever import Retriever
from .embedder import Embedder
from config import OLLAMA_HOST, OLLAMA_API_GENERATE_URL, MODEL
import httpx
import os
from typing import AsyncGenerator
import json
from logging_config import get_logger, Timer, DEBUG

logger = get_logger(__name__)
logger.setLevel(DEBUG)
embedder = Embedder()

class Generator:
    def __init__(self, model: str | None = None): 
        self.client = httpx.AsyncClient(
            timeout=None,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100)
        )
        self.model = model or MODEL
        logger.info(
            "HTTP client initialized"
        )
    def set_model(self, model: str):
        self.model = model
        logger.debug(
            "model set successfully"
        )

    async def get_retrieved_chunks(self, doc_id: str, query: str) -> list[str] | str:
        retriever = Retriever(doc_id, embedder)
        try:
            retrieved_chunks = await retriever.retrieve(query=query)
            return retrieved_chunks
        except Exception:
            logger.error(
                "could not get retrieved chunks"
            )
            raise
    
    async def generate_stream(self, doc_id: str, query: str) -> AsyncGenerator[str, None]:
        retrieved_chunks = await self.get_retrieved_chunks(doc_id=doc_id, query=query)
        request_body = {
            "model": self.model,
            "prompt": f"Generate response using only these text chunks: {retrieved_chunks}.\n Query: {query}",
            "stream": True,
            "keep_alive": "10m"
        }
        async with Timer() as timer:
            async with self.client.stream("POST", OLLAMA_API_GENERATE_URL, json=request_body) as response:
                try:
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
                            "model": self.model,
                        }
                    )
                except Exception as e:
                    logger.error(
                        "could not stream response"
                    )
                    raise e
    async def ollama_check(self) -> bool:
        try:
            response = await self.client.post(f"{OLLAMA_HOST}/api/show", json={
                "model": self.model
            })

            if response.status_code != 200:
                logger.error(
                    "Ollama model not found",
                    extra={
                        "ollama_host": OLLAMA_HOST,
                        "ollama_model": self.model,
                        "status_code": response.status_code,
                        "response_text": response.text
                    }
                )
                if self.model != MODEL:
                    self.set_model(MODEL)
                    logger.debug(
                        f"using {MODEL} instead, re-running the health check..."
                    )
                    await self.ollama_check()
                else:
                    logger.critical("even the default model ain't available")
                    raise Exception("models unavailable")

            logger.debug(
                "Ollama API running",
                extra={
                    "ollama_host": OLLAMA_HOST,
                    "model": self.model,
                }
            )
            return True
        except Exception as e:
            logger.critical(
                f"Error during Ollama health check [{type(e).__name__}]"
            )
            os._exit(1)

    async def close(self):
        await self.client.aclose()