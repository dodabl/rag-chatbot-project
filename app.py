from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rag import Generator
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
GENERATOR: Generator = Generator()
MANAGER: FileManager = FileManager()
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await GENERATOR.ollama_check()

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

app = FastAPI(lifespan=lifespan)

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

@app.get("/health")
async def health_check():
    try:
        await GENERATOR.ollama_check()
        return {"status_code": 200}
    except Exception as e:
        raise e
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
            logger.error(
                f"missing meta.json file in directory [{element.name}]"
            )
            continue
        logger.info(
            "Documents loading complete"
        )
    return {
        "documents": docs
    }

@app.get("/chat/{doc_id}/history")
async def get_chat_history(doc_id: str) -> dict[str, list[Message]]:
    history = CHAT_HISTORY_DIR / f"{doc_id}.json"
    try:
        async with aiofiles.open(history, "r") as f:
            content = await f.read()
            data: list[Message] = json.loads(content).get("history", [])
            logger.info(
                "chat history loaded with success",
                extra={
                    "doc_id": doc_id
                }
            )
            return {"history": data}
    except json.JSONDecodeError as e:
        data: list[Message] = []
        logger.error(
            f"error during chat history loading: [{e}]"
        )
        raise e
    except FileNotFoundError as e:
        logger.error(
            "chat history file not found",
            extra={
                "doc_id": doc_id
            }
        )
        raise e
    except Exception as e:
        logger.error(
            "error during chat history loading",
            extra={
                "doc_id": doc_id
            }
        )
        raise e



@app.post("/upload")
async def upload_file(file: UploadFile) -> dict[str, bool | str]:
    if not file.filename:
        return {"status": "error"}
    doc_id, file_path, index_dir = await MANAGER.store_file(file)
    chunks = index_dir / "chunks.json"
    index = index_dir / "index.faiss"
    try:
        async with aiofiles.open(index_dir / "meta.json", "r") as f:
            file_meta = json.loads(await f.read())
        logger.info(
            "file meta.json read with success"
        )
        result = {
                "status": "ok",
                "doc_id": doc_id,
                "filename": file_meta["filename"],
                "file-path": str(file_path),
                "chunks-path": str(chunks),
                "index-path": str(index)
                }
    except Exception as e:
        logger.error(
            "could not read file: meta.json"
        )
        result = {
            "status": "error",
            "doc_id": doc_id,
            "file_path": file_path,
            "chunks-path": str(chunks),
            "index-path": str(index)
        }
    finally:
        return result



@app.post("/chat/{doc_id}")
async def ask_question(doc_id: str, question: Question) -> StreamingResponse:
    async def event_stream():
        try:
            async for chunk in GENERATOR.generate_stream(doc_id, question.question):
                yield chunk
        except Exception as e:
            logger.error(
                f"error during streaming: {e}"
            )
            yield "\n\n[error: the response was interrupted. please try again.]"
    return StreamingResponse(event_stream(), media_type="text/plain")

@app.post("/chat/{doc_id}/history")
async def update_chat_history(doc_id: str,message: Message):
    history_file = CHAT_HISTORY_DIR / f"{doc_id}.json"
    msg_json = {
        "role": message.role,
        "text": message.text
    }
    try:
        async with aiofiles.open(history_file, "r") as f:
            history = json.loads(await f.read())
    except Exception:
        logger.error(
            "could not read chat history file"
        )
        history = {"history": []}
    finally:
        history.get("history", []).append(msg_json)
        async with aiofiles.open(history_file, "w") as f:
            await f.write(json.dumps(history))
        return {"status": "ok"}
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static-assets")