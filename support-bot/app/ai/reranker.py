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
Твоя задача — отобрать самые релевантные чанки и оценить, насколько полно они отвечают на вопрос.
Всегда возвращай в selected_indices 1-3 самых релевантных чанка, если в списке вообще есть хоть один чанк по теме вопроса (тот же раздел/метрика/сущность). Не оставляй selected_indices пустым только из-за того, что чанк не содержит дословного исчерпывающего ответа — частично релевантный, но тематически верный чанк всё равно полезен ассистенту.
Выбирай 1-2 самых прямых чанка. Третий чанк выбирай только если он так же прямо относится к теме вопроса.
Не выбирай общий, соседний или похожий контекст, если он относится к ДРУГОЙ метрике, отчету или разделу.
Строго разделяй метрики: продажи, выплаты, реклама, прибыль и налоговая база — это разные темы.
Если пользователь спрашивает про продажи, не выбирай чанки про сумму выплат, кроме случаев, когда чанк явно объясняет сверку продаж.
Если пользователь спрашивает про выплаты, не выбирай чанки про рекламу как "дополнение", кроме случаев, когда вопрос прямо про рекламные удержания.
Если пользователь спрашивает про налоговую базу, выбирай только чанки про налоги, налоговую базу или сверку данных по налогам/отчетам.
Флаг enough_context оценивает ПОЛНОТУ выбранных чанков:
- enough_context=true — выбранные чанки прямо и полно отвечают на вопрос.
- enough_context=false — релевантные по теме чанки есть, но ответа в них недостаточно или он неполный. selected_indices при этом всё равно заполни лучшими чанками — финальный ассистент сам решит, ответить частично или вызвать оператора.
Оставляй selected_indices пустым (и enough_context=false) ТОЛЬКО когда ни один чанк не относится к теме вопроса или все чанки относятся к другим разделам/метрикам.
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
    params = {
        "model": settings.openai_rerank_model or settings.openai_model,
        "instructions": _RERANK_INSTRUCTIONS.strip(),
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rag_rerank_result",
                "schema": _RERANK_SCHEMA,
                "strict": True,
            },
        },
    }
    if settings.openai_reasoning_effort is not None:
        params["reasoning"] = {"effort": settings.openai_reasoning_effort}
    response = await _client.responses.create(**params)
    data = json.loads(response.output_text or "{}")
    selected = [
        index
        for index in data.get("selected_indices", [])
        if isinstance(index, int) and 0 <= index < len(chunks)
    ][: settings.rag_rerank_keep]
    enough_context = bool(data.get("enough_context"))
    # Не обнуляем selected при enough_context=false: частично релевантные чанки
    # передаём ассистенту, а решение "звать оператора" принимает он сам.
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
