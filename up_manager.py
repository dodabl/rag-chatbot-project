from pathlib import Path
import os
from fastapi import UploadFile
from config import UPLOADS_DIR, VECTOR_STORE_DIR, CHAT_HISTORY_DIR
from rag.ingest import PDFIngestor
import hashlib
import aiofiles
import json
import datetime

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
        file_content = await file.read()
        exists = False
        file_id = self.get_file_id(file_content)
        file_path = UPLOADS_DIR / f"{file_id}.pdf"
        index_dir = VECTOR_STORE_DIR / file_id
        history_path = CHAT_HISTORY_DIR / f"{file_id}.json"
        if file_path.is_file():
            exists = True
            return exists, file_id, file_path, index_dir
        meta_path = index_dir / "meta.json"
        os.makedirs(index_dir, exist_ok=True)
        ingestor = PDFIngestor(file_id)
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(file_content)
        meta_dict = self.get_meta_dict(file)
        await ingestor.ingest()
        async with aiofiles.open(meta_path, "w") as f:
            await f.write(json.dumps(meta_dict))
        async with aiofiles.open(history_path, "wb") as f:
            await f.write(json.dumps({"history": []}).encode("utf-8"))
        return exists, file_id, file_path, index_dir