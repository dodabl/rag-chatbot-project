from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag.generator import Generator
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import BASE_DIR, VECTOR_STORE_DIR
from up_manager import FileManager 
import json
import aiofiles
import datetime

GENERATOR = Generator()
MANAGER = FileManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await GENERATOR.close()

class Question(BaseModel):
    question: str

class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    uploaded_at: datetime.datetime


app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/documents")
async def get_documents()-> dict[str, list[DocumentSummary]]:
    docs: list[DocumentSummary] = []
    it = VECTOR_STORE_DIR.iterdir()
    for element in it:
        if not element.is_dir():
            continue
        meta = element / "meta.json"
        try:
            async with aiofiles.open(meta, 'r') as f:
                meta_dict = json.loads(await f.read())
            doc = DocumentSummary(
                doc_id=element.name,
                filename=meta_dict["filename"],
                uploaded_at=meta_dict["uploaded_at"]
            )
            docs.append(doc)
        except Exception:
            print(f"[ERROR] missing meta.json file in directory [{element.name}]")
            continue
    return {
        "documents": docs
    }




@app.post("/upload")
async def upload_file(file: UploadFile) -> dict[str, bool | str]:
    if not file.filename:
        return {"status": "error"}
    exists, file_path, index_dir = await MANAGER.store_file(file)
    chunks = index_dir / "chunks.json"
    index = index_dir / "index.faiss"
    async with aiofiles.open(index_dir / "meta.json", "r") as f:
        file_meta = json.loads(await f.read())
    return {
        "status": "ok",
        "exists": exists,
        "filename": file_meta["filename"],
        "file-path": str(file_path),
        "chunks-path": str(chunks),
        "index-path": str(index)
        }
@app.post("/chat")
async def ask_question(question: Question) -> StreamingResponse:
    async def event_stream():
        try:
            async for chunk in GENERATOR.generate_stream(question.question):
                yield chunk
        except Exception as e:
            print(f"error during streaming: {e}")
            yield "\n\n[error: the response was interrupted. please try again.]"
    return StreamingResponse(event_stream(), media_type="text/plain")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static-assets")
app.mount("/", StaticFiles(directory=str(BASE_DIR / "templates"), html=True), name="static")