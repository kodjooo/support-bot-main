import math

from app.embeddings import EmbeddingResult, EmbeddingService, chunk_text


def test_chunk_text_with_overlap() -> None:
    text = " ".join(f"word{i}" for i in range(10))

    chunks = chunk_text(text, max_tokens=4, overlap=2)

    assert chunks == [
        "word0 word1 word2 word3",
        "word2 word3 word4 word5",
        "word4 word5 word6 word7",
        "word6 word7 word8 word9",
    ]


def test_chunk_text_empty() -> None:
    assert chunk_text("", max_tokens=4, overlap=1) == []


class FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls = []
        self.embeddings = self

    def create(self, model: str, input: str):
        index = len(self.calls)
        vector = [float(index), math.pi]
        self.calls.append((model, input))
        entry = type("Entry", (), {"embedding": vector})
        return type("Resp", (), {"data": [entry()]})()


class _Secret:
    def get_secret_value(self) -> str:
        return "key"


class _Settings:
    openai_api_key = _Secret()
    openai_embedding_model = "text-embedding-3-small"


def test_embedding_service_generates_vectors() -> None:
    settings = _Settings()

    client = FakeEmbeddingsClient()
    service = EmbeddingService(settings=settings, client=client, retry_attempts=1)
    texts = ["первый", "", "второй"]

    result = service.embed_texts(texts)

    assert len(result) == 2
    assert isinstance(result[0], EmbeddingResult)
    assert client.calls[0][1] == "первый"
