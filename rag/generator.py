from retriever import Retriever
import requests
API_GENERATE_URL = "http://localhost:11434/api/generate"

class Generator:
    def __init__(self, chunks_path: str = "../data/vector_store/chunks.json"):
        self.chunks_path = chunks_path
    def get_retrieved_chunks(self, query: str) -> list[str]:
        _retriever = Retriever(chunks_path=self.chunks_path)
        retrieved_chunks = _retriever.retrieve(query=query)
        return retrieved_chunks
    def get_response(self, query: str) -> requests.Response:
        retrieved_chunks = self.get_retrieved_chunks(query=query)
        request_body = {
            "model": "llama3",
            "prompt": f"Generate response using only these text chunks: {retrieved_chunks}.\nprompt: {query}",
            "stream": False
        }
        response = requests.post(API_GENERATE_URL, json=request_body)
        return response
    def generate(self, query: str) -> str:
        response = self.get_response(query=query)
        return response.json()["response"]