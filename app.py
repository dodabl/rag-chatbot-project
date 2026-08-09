from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from rag.ingest import PDFIngestor
from rag.generator import Generator
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import UPLOADS_DIR
import aiofiles

GENERATOR = Generator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await GENERATOR.close()

class Question(BaseModel):
    question: str
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile) -> dict[str, str]:
    if not file.filename:
        return {"status": "error"}
    upload_path = UPLOADS_DIR / file.filename
    async with aiofiles.open(upload_path, "wb") as f:
        await f.write(await file.read())
        await f.close()
    ingestor = PDFIngestor(upload_path)
    chunks, embeddings, index = await ingestor.ingest()

    
    return {
        "status": "ok",
        "filename": f"{file.filename}",
        "file-path": str(upload_path),
        "chunks-path": str(chunks),
        "embeddings-path": str(embeddings),
        "index-path": str(index)
        }
@app.post("/chat")
async def ask_question(question: Question) -> StreamingResponse:
    async def event_stream():
        async for chunk in GENERATOR.generate_stream(question.question):
            yield chunk
    return StreamingResponse(event_stream(), media_type="text/plain")