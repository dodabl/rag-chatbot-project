import pytest
from . import Embedder



@pytest.fixture(scope="module")
def embedder():
    _embedder = Embedder()
    yield _embedder


@pytest.mark.parametrize("texts", (
    "abcd 124657q9 whgd 2387b iusdy23",
    ["dhad", "uhufe64r78", "jhsd89273jhd327 jdh9237"],
    ["", "  ", "   ", "text with content"]
))
@pytest.mark.asyncio
async def test_encode_string(embedder, texts):
    embedding = await embedder.encode(texts)
    assert embedding.shape[1] == 384

@pytest.mark.parametrize("texts", (
    "",
    ["", "  "]
))

@pytest.mark.asyncio
async def test_encode_empty_string(embedder, texts):
    with pytest.raises(ValueError):
        await embedder.encode(texts)