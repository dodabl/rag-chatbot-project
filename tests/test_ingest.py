import pytest
from . import PDFIngestor
from pathlib import Path
import os
import numpy as np

@pytest.fixture(scope="module")
def ingestor():
    os.chdir(Path(__file__).parent)
    yield PDFIngestor()
    os.chdir(Path(os.getcwd()).parent)


@pytest.mark.asyncio
async def test_extract_chunks(ingestor, monkeypatch):

    monkeypatch.setattr(ingestor, "path", Path(f"fixtures/test_sample.pdf"))

    chunks = await ingestor.extract_chunks()
    print(f"chunks: {chunks}")
    assert len(chunks) > 0

@pytest.mark.asyncio
async def test_extract_chunks_empty_file(ingestor, monkeypatch):
    monkeypatch.setattr(ingestor, "path", Path(f"fixtures/test_blank.pdf"))

    with pytest.raises(Exception):
        await ingestor.extract_chunks()


@pytest.mark.parametrize("id",(
    "",
    "abcd"
))
@pytest.mark.asyncio
async def test_extract_chunks_invalid_path(ingestor, id, monkeypatch):
    id = id.strip()
    monkeypatch.setattr(ingestor, "path", Path(f"fixtures/{id}.pdf"))

    with pytest.raises(FileNotFoundError):
        await ingestor.extract_chunks()


@pytest.mark.parametrize("embeddings", (
    np.ones((1, 384)),
    np.ones((5, 384))
))
@pytest.mark.asyncio
async def test_create_faiss_index(ingestor, embeddings):
    index = await ingestor.create_faiss_index(embeddings)
    assert index.ntotal == embeddings.shape[0]

@pytest.mark.parametrize("embeddings", (
    np.ones((0, 384)),
    np.ones((1, 5))
))
@pytest.mark.asyncio
async def test_create_faiss_index_invalid_shape(ingestor, embeddings):
    embeddings = np.ones((0, 384))
    with pytest.raises(ValueError):
        await ingestor.create_faiss_index(embeddings)