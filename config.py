from pathlib import Path
import os

BASE_DIR: Path = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"
CHAT_HISTORY_DIR = BASE_DIR / "data" / "chat_history"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_API_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
MODEL = "llama3.2:3b"