from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.ai.openai_client import create_async_openai_client
from app.config import settings

logger = logging.getLogger(__name__)

_client = create_async_openai_client()


@dataclass
class PlanningResult:
    status: str
    clarifying_question: str | None
    search_query: str | None
    extracted: dict
    confidence: float

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and bool((self.search_query or "").strip())


_PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ready", "need_clarification"]},
        "clarifying_question": {"type": ["string", "null"]},
        "search_query": {"type": ["string", "null"]},
        "extracted": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "marketplace": {"type": ["string", "null"]},
                "section": {"type": ["string", "null"]},
                "metric": {"type": ["string", "null"]},
                "period": {"type": ["string", "null"]},
                "problem": {"type": ["string", "null"]},
            },
            "required": ["marketplace", "section", "metric", "period", "problem"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["status", "clarifying_question", "search_query", "extracted", "confidence"],
}

_PLANNER_INSTRUCTIONS = """
Ты внутренний planner RAG-бота поддержки Sellerdata. Не отвечай пользователю.
Определи, достаточно ли пользовательского запроса для точного поиска в базе знаний.

Политика уточнений умеренная:
- Если запрос связан с Sellerdata/Wildberries/Ozon и понятны раздел, метрика или конкретная проблема, верни ready.
- Если короткий запрос может относиться к разным разделам, отчетам или метрикам, верни need_clarification.
- Уточняющий вопрос должен быть один, короткий, на русском, без markdown.
- search_query должен быть нормализованным запросом для базы знаний: marketplace, раздел, метрика, проблема, период, если они известны.
- Не придумывай неизвестные значения extracted: ставь null.
"""


async def plan_query(
    user_texts: list[str],
    *,
    pending: dict | None = None,
) -> PlanningResult:
    """Планирует RAG-поиск или уточняющий вопрос. Stateless, без previous_response_id."""
    payload = {
        "current_user_texts": user_texts,
        "pending_clarification": pending,
    }
    response = await _client.responses.create(
        model=settings.openai_planner_model or settings.openai_model,
        instructions=_PLANNER_INSTRUCTIONS.strip(),
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_planning_result",
                "schema": _PLANNER_SCHEMA,
                "strict": True,
            },
        },
    )
    data = json.loads(response.output_text or "{}")
    result = PlanningResult(
        status=data["status"],
        clarifying_question=data.get("clarifying_question"),
        search_query=_normalize_search_query(data, user_texts),
        extracted=data.get("extracted") or {},
        confidence=float(data.get("confidence") or 0),
    )
    logger.info(
        "[PLANNER] status=%s confidence=%s search_query=%s question=%s",
        result.status,
        result.confidence,
        (result.search_query or "")[:300],
        result.clarifying_question,
    )
    return result


def _normalize_search_query(data: dict, user_texts: list[str]) -> str | None:
    """Преобразует служебные key:value варианты модели в естественную поисковую фразу."""
    query = (data.get("search_query") or "").strip()
    if not query:
        return None
    if not any(marker in query for marker in (":", ";", "|")):
        return query

    extracted = data.get("extracted") or {}
    parts = [
        str(value).strip()
        for key in ("marketplace", "section", "metric", "problem", "period")
        for value in [extracted.get(key)]
        if value and str(value).strip().lower() != "null"
    ]
    original = " ".join(text.strip() for text in user_texts if text.strip())
    if original:
        parts.append(original)
    if parts:
        return " ".join(dict.fromkeys(parts))
    return query.replace(";", " ").replace("|", " ").replace(":", " ")
