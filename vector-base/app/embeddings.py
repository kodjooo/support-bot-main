from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def chunk_text(text: str, *, max_tokens: int, overlap: int) -> List[str]:
    """Разбивает текст на чанки по словам, допускает перекрытие."""
    if max_tokens <= 0:
        raise ValueError("max_tokens должен быть положительным.")
    if overlap < 0:
        raise ValueError("overlap не может быть отрицательным.")
    if overlap >= max_tokens:
        raise ValueError("overlap должен быть меньше max_tokens.")

    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    total = len(words)

    while start < total:
        end = min(start + max_tokens, total)
        chunks.append(" ".join(words[start:end]))

        if end >= total:
            break

        start = end - overlap if overlap else end

    return chunks


@dataclass
class EmbeddingResult:
    text: str
    embedding: Sequence[float]


class EmbeddingService:
    """Сервис генерации эмбеддингов OpenAI с повторами."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: OpenAI | None = None,
        retry_attempts: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or OpenAI(api_key=self.settings.openai_api_key.get_secret_value())

        attempts = retry_attempts or 3
        self._retryer = Retrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    def embed_texts(self, texts: Iterable[str]) -> List[EmbeddingResult]:
        results: List[EmbeddingResult] = []
        for text in texts:
            if not text:
                logger.debug("Пропущен пустой фрагмент при генерации эмбеддингов.")
                continue

            embedding = self._retryer(self._create_embedding, text)
            results.append(EmbeddingResult(text=text, embedding=embedding))
        return results

    def _create_embedding(self, text: str) -> Sequence[float]:
        logger.debug("Запрос к OpenAI Embeddings для текста длиной %s символов.", len(text))
        response = self.client.embeddings.create(
            model=self.settings.openai_embedding_model,
            input=text,
        )
        return response.data[0].embedding
