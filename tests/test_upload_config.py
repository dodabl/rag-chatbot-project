import pytest
import os
from pathlib import Path
from . import FileManager
from fastapi import UploadFile

@pytest.fixture(scope="module")
def upload_config():
    os.chdir(Path(__file__).parent)
    yield FileManager()
    os.chdir(Path(os.getcwd()).parent)


@pytest.mark.asyncio
async def test_get_file_id(upload_config):
    file_content = b"this is supposed to be a pdf file content"
    file_id = upload_config.get_file_id(file_content)
    assert isinstance(file_id, str)
    assert len(file_id) == 12

@pytest.mark.asyncio
async def test_get_file_id_empty_content(upload_config):
    file_content = b""
    with pytest.raises(ValueError):
        upload_config.get_file_id(file_content)


def test_get_meta_dict(upload_config):
    file = UploadFile(filename="test.pdf", file=open("fixtures/test_sample.pdf", "rb"))
    meta_dict = upload_config.get_meta_dict(file)
    assert isinstance(meta_dict, dict)
    assert "filename" in meta_dict
    assert "size" in meta_dict
    assert "uploaded_at" in meta_dict

