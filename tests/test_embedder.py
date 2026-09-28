import pytest
from . import Embedder



@pytest.fixture(scope="module")
def setup():
    embedder = Embedder()
    yield embedder


@pytest.mark.parametrize("texts", (
    "",
    "abcd 124657q9 whgd 2387b iusdy23",
    ["dhad", "uhufe64r78", "jhsd89273jhd327 jdh9237"]
))


@pytest.mark.asyncio
async def test_encode(setup, texts):
    embedder = setup
    embedding = await embedder.encode(texts)
    assert embedding.shape[1] == 384