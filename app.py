from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/upload/")
async def upload_file(file: UploadFile) -> dict[str, str]:
    with open(f"data/uploads/{file.filename}", "wb") as f:
        f.write(await file.read())
        f.close()
    return {
        "status": "ok",
        "filename": f"{file.filename}"
        }