import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")

from app.ai.planner import plan_query


def _response(payload: dict):
    resp = MagicMock()
    resp.output_text = json.dumps(payload, ensure_ascii=False)
    return resp


@pytest.mark.asyncio
async def test_planner_returns_need_clarification():
    payload = {
        "status": "need_clarification",
        "clarifying_question": "Где именно не сходятся продажи?",
        "search_query": None,
        "extracted": {
            "marketplace": None,
            "section": None,
            "metric": "продажи",
            "period": None,
            "problem": "расхождение",
        },
        "confidence": 0.4,
    }
    with patch("app.ai.planner._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=_response(payload))
        result = await plan_query(["у меня не сходятся продажи"])

    assert result.status == "need_clarification"
    assert result.clarifying_question == "Где именно не сходятся продажи?"
    assert result.is_ready is False
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert "previous_response_id" not in call_kwargs


@pytest.mark.asyncio
async def test_planner_returns_ready_search_query():
    payload = {
        "status": "ready",
        "clarifying_question": None,
        "search_query": "расхождение продаж Wildberries в Дашборде Sellerdata",
        "extracted": {
            "marketplace": "Wildberries",
            "section": "Дашборд",
            "metric": "продажи",
            "period": None,
            "problem": "расхождение",
        },
        "confidence": 0.9,
    }
    with patch("app.ai.planner._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=_response(payload))
        result = await plan_query(["В Дашборде не сходятся продажи WB"])

    assert result.is_ready is True
    assert result.search_query == "расхождение продаж Wildberries в Дашборде Sellerdata"


@pytest.mark.asyncio
async def test_planner_normalizes_structured_search_query():
    payload = {
        "status": "ready",
        "clarifying_question": None,
        "search_query": "marketplace: Wildberries; section: Дашборд; metric: Сумма продаж",
        "extracted": {
            "marketplace": "Wildberries",
            "section": "Дашборд",
            "metric": "Сумма продаж",
            "period": "прошлая неделя",
            "problem": "несовпадение с еженедельным отчетом",
        },
        "confidence": 0.9,
    }
    with patch("app.ai.planner._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=_response(payload))
        result = await plan_query(["В Дашборде не сходятся продажи WB"])

    assert ":" not in result.search_query
    assert ";" not in result.search_query
    assert "Wildberries" in result.search_query
    assert "В Дашборде не сходятся продажи WB" in result.search_query
