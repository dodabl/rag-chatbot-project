from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent

UPLOADS_DIR = BASE_DIR / "data" / "uploads"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_API_GENERATE_URL = "http://localhost:11434/api/generate"