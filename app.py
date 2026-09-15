from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag.generator import Generator
from pydantic import BaseModel
from contextlib import asynccontextmanager
from config import BASE_DIR, VECTOR_STORE_DIR, CHAT_HISTORY_DIR
from upload_config import FileManager 
import json
import aiofiles
import datetime
from fastapi.templating import Jinja2Templates
from logging_config import setup_logging, get_logger, Timer
import uuid

setup_logging()
logger = get_logger(__name__)

GENERATOR = Generator()
MANAGER = FileManager()
templates = Jinja2Templates(directory="templates")

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

class Message(BaseModel):
    role: str | None
    text: str | None

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    async with Timer() as timer:
        response = await call_next(request)
    logger.info(
        "Request handled",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": request.url.path,
            "status_code": response.status_code,
            "duration_ms": timer.elapsed_ms
        }
    )
    return response

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/documents")
async def get_documents()-> dict[str, list[DocumentSummary]]:
    docs: list[DocumentSummary] = []
    itr = VECTOR_STORE_DIR.iterdir()
    for element in itr:
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

@app.get("/chat/{doc_id}/history")
async def get_chat_history(doc_id: str) -> dict[str, list[Message]]:
    history = CHAT_HISTORY_DIR / f"{doc_id}.json"
    if history.is_file() and history.stat().st_size > 0:
        async with aiofiles.open(history, "r") as f:
            content = await f.read()
        try:
            data: list[Message] = json.loads(content)["history"]
        except json.JSONDecodeError:
            data: list[Message] = []  # or {} — match whatever your actual stored shape is
    else:
        data = []
    return {"history": data}



@app.post("/upload")
async def upload_file(file: UploadFile) -> dict[str, bool | str]:
    if not file.filename:
        return {"status": "error"}
    exists, doc_id, file_path, index_dir = await MANAGER.store_file(file)
    chunks = index_dir / "chunks.json"
    index = index_dir / "index.faiss"
    async with aiofiles.open(index_dir / "meta.json", "r") as f:
        file_meta = json.loads(await f.read())
    return {
        "status": "ok",
        "exists": exists,
        "doc_id": doc_id,
        "filename": file_meta["filename"],
        "file-path": str(file_path),
        "chunks-path": str(chunks),
        "index-path": str(index)
        }
@app.post("/chat/{doc_id}")
async def ask_question(doc_id: str, question: Question) -> StreamingResponse:
    async def event_stream():
        try:
            async for chunk in GENERATOR.generate_stream(doc_id, question.question):
                yield chunk
        except Exception as e:
            print(f"error during streaming: {e}")
            yield "\n\n[error: the response was interrupted. please try again.]"
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.post("/chat/{doc_id}/history")
async def update_chat_history(doc_id: str,message: Message):
    history_file = CHAT_HISTORY_DIR / f"{doc_id}.json"
    msg_json = {
        "role": message.role,
        "text": message.text
    }
    async with aiofiles.open(history_file, "r") as f:
        history = json.loads(await f.read())
    history["history"].append(msg_json)
    async with aiofiles.open(history_file, "w") as f:
        await f.write(json.dumps(history))
    return {"status": "ok"}
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static-assets")