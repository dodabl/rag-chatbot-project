from pathlib import Path
import os
from fastapi import UploadFile
from config import UPLOADS_DIR, VECTOR_STORE_DIR, CHAT_HISTORY_DIR
from rag.ingest import PDFIngestor
import hashlib
import aiofiles
import json
import datetime
from logging_config import get_logger, Timer
logger = get_logger(__name__)

class FileManager():
    def __init__(self):
        pass
    def get_file_id(self, file_content: bytes) -> str:
        return hashlib.sha256(file_content).hexdigest()[:12]
    def get_meta_dict(self, file: UploadFile) -> dict[str, datetime.datetime | int | str]:
        meta_dict = {
            "filename": file.filename,
            "size": file.size,
            "uploaded_at": str(datetime.datetime.now())
        }
        return meta_dict

    async def store_file(self, file: UploadFile)-> tuple[bool, str, Path, Path]:
        async with Timer() as timer:
            file_content = await file.read()
            logger.info(
                "file read from upload",
                extra={
                    "pdfname": file.filename,
                    "size": len(file_content)
                }
            )

            exists = False
            file_id = self.get_file_id(file_content)
            file_path = UPLOADS_DIR / f"{file_id}.pdf"
            index_dir = VECTOR_STORE_DIR / file_id
            history_path = CHAT_HISTORY_DIR / f"{file_id}.json"
            if file_path.is_file():
                exists = True
                logger.info(
                    "file already exists",
                    extra={
                        "file_id": file_id,
                        "file_path": str(file_path),
                        "index_dir": str(index_dir),   
                    }
                )
                return exists, file_id, file_path, index_dir
            meta_path = index_dir / "meta.json"
            os.makedirs(index_dir, exist_ok=True)
            ingestor = PDFIngestor(file_id)
            async with aiofiles.open(file_path, "wb") as f:
                await f.write(file_content)
            logger.info(
                "file written to disk",
                extra={
                    "file_path": str(file_path),
                    "index_dir": str(index_dir)
                }
            )

            meta_dict = self.get_meta_dict(file)
            await ingestor.ingest()
            async with aiofiles.open(meta_path, "w") as f:
                await f.write(json.dumps(meta_dict))
            logger.info(
                "meta.json written to disk",
                extra={
                    "meta_path": str(meta_path)
                }
            )
            if not history_path.is_file():
                async with aiofiles.open(history_path, "wb") as f:
                    await f.write(json.dumps({"history": []}).encode("utf-8"))
                logger.info(
                    "chat history initialized",
                    extra={
                        "history_path": str(history_path)
                    }
                )
        logger.info(
            "file upload",
            extra={
                "file_id": file_id,
                "file_path": str(file_path),
                "index_dir": str(index_dir),
                "duration_ms": timer.elapsed_ms
            }
        )
        return exists, file_id, file_path, index_dir