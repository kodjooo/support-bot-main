from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.ai.openai_client import create_async_openai_client
from app.ai.vector_client import ContextChunk
from app.config import settings

logger = logging.getLogger(__name__)

_client = create_async_openai_client()


@dataclass
class RerankResult:
    enough_context: bool
    selected_indices: list[int]
    reason: str


_RERANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "enough_context": {"type": "boolean"},
        "selected_indices": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
        },
        "reason": {"type": "string"},
    },
    "required": ["enough_context", "selected_indices", "reason"],
}

_RERANK_INSTRUCTIONS = """
Ты внутренний reranker RAG-бота поддержки Sellerdata. Не отвечай пользователю.
Выбери только те чанки, которые прямо помогают ответить на исходный вопрос.
Не выбирай общий, соседний или похожий контекст, если он относится к другой метрике, отчету или разделу.
Выбирай 1-2 самых прямых чанка. Третий чанк выбирай только если он так же прямо отвечает на вопрос, а не просто дополняет тему.
Не выбирай косвенные "возможные причины", обзорные описания раздела или соседние инструкции, если есть более прямой чанк.
Строго разделяй метрики: продажи, выплаты, реклама, прибыль и налоговая база — это разные темы.
Если пользователь спрашивает про продажи, не выбирай чанки про сумму выплат, кроме случаев, когда чанк явно объясняет сверку продаж.
Если пользователь спрашивает про выплаты, не выбирай чанки про рекламу как "дополнение", кроме случаев, когда вопрос прямо про рекламные удержания.
Если пользователь спрашивает про налоговую базу, выбирай только чанки про налоги, налоговую базу или сверку данных по налогам/отчетам.
Лучше верни fewer selected_indices или enough_context=false, чем смешай нерелевантные разделы.
Если подходящего контекста нет или чанки противоречат друг другу, верни enough_context=false.
Индексы selected_indices должны быть 0-based индексами из списка chunks.
"""


async def rerank_context(
    *,
    user_query: str,
    search_query: str,
    chunks: list[ContextChunk],
) -> RerankResult:
    """Отбирает релевантные чанки для финального ответа. Stateless, без previous_response_id."""
    payload = {
        "user_query": user_query,
        "search_query": search_query,
        "max_selected": settings.rag_rerank_keep,
        "chunks": [
            {
                "index": index,
                "text": chunk.text,
                "score": chunk.score,
                "distance": chunk.distance,
                "metadata": chunk.metadata or {},
                "matched_terms": chunk.matched_terms or [],
            }
            for index, chunk in enumerate(chunks)
        ],
    }
    response = await _client.responses.create(
        model=settings.openai_rerank_model or settings.openai_model,
        instructions=_RERANK_INSTRUCTIONS.strip(),
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_rerank_result",
                "schema": _RERANK_SCHEMA,
                "strict": True,
            },
        },
    )
    data = json.loads(response.output_text or "{}")
    selected = [
        index
        for index in data.get("selected_indices", [])
        if isinstance(index, int) and 0 <= index < len(chunks)
    ][: settings.rag_rerank_keep]
    enough_context = bool(data.get("enough_context"))
    if not enough_context:
        selected = []
    result = RerankResult(
        enough_context=enough_context,
        selected_indices=selected,
        reason=str(data.get("reason") or ""),
    )
    logger.info(
        "[RERANK] enough_context=%s selected=%s reason=%s",
        result.enough_context,
        result.selected_indices,
        result.reason[:300],
    )
    return result
