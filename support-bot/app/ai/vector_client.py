"""HTTP-клиент для поиска релевантных чанков в vector-base."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Таймаут запроса к сервису векторного поиска (секунды)
_REQUEST_TIMEOUT = 5.0


@dataclass
class ContextChunk:
    """Фрагмент контекста из базы знаний."""

    text: str
    score: float | None = None
    distance: float | None = None
    metadata: dict | None = None
    matched_terms: list[str] | None = None

    def to_prompt_text(self) -> str:
        metadata = self.metadata or {}
        title = metadata.get("title") or metadata.get("section") or "без названия"
        return f"Источник: {title}\nФрагмент: {self.text}"


async def fetch_context(query: str) -> list[str]:
    """
    Возвращает релевантные чанки из векторной базы знаний.
    При недоступности сервиса или ошибке возвращает пустой список —
    бот продолжит работу без контекста.
    """
    if not settings.vector_base_url:
        return []

    url = f"{settings.vector_base_url.rstrip('/')}/search"

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            logger.info("[VECTOR] Запрос поиска: %s", query[:300])
            response = await client.post(url, json={"query": query, "top_k": 3})
            response.raise_for_status()
            data = response.json()
            chunks = _parse_context(data)
            logger.info("[VECTOR] Найдено чанков: %d", len(chunks))
            for i, chunk in enumerate(chunks):
                logger.info(
                    "[VECTOR] Чанк %d: score=%s distance=%s metadata=%s terms=%s text=%s",
                    i + 1,
                    chunk.score,
                    chunk.distance,
                    chunk.metadata,
                    chunk.matched_terms,
                    chunk.text[:200],
                )
            return [chunk.to_prompt_text() for chunk in chunks]
    except httpx.TimeoutException:
        logger.warning("Таймаут запроса к векторной базе (%s).", url)
    except httpx.HTTPStatusError as exc:
        logger.warning("Ошибка HTTP от векторной базы: %s.", exc.response.status_code)
    except Exception:
        logger.warning("Не удалось получить контекст из векторной базы.", exc_info=True)

    return []


def _parse_context(data: dict) -> list[ContextChunk]:
    results = data.get("results") or []
    if results:
        return [
            ContextChunk(
                text=item.get("text") or "",
                score=item.get("score"),
                distance=item.get("distance"),
                metadata=item.get("metadata") or {},
                matched_terms=item.get("matched_terms") or [],
            )
            for item in results
            if item.get("text")
        ]

    return [ContextChunk(text=chunk) for chunk in data.get("chunks") or [] if chunk]
