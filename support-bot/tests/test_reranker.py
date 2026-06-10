import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")

from app.ai.reranker import _RERANK_INSTRUCTIONS, rerank_context
from app.ai.vector_client import ContextChunk


def _response(payload: dict):
    resp = MagicMock()
    resp.output_text = json.dumps(payload, ensure_ascii=False)
    return resp


def test_reranker_instructions_separate_metrics():
    assert "Строго разделяй метрики" in _RERANK_INSTRUCTIONS
    assert "продажи, выплаты, реклама, прибыль и налоговая база" in _RERANK_INSTRUCTIONS
    assert "не выбирай чанки про сумму выплат" in _RERANK_INSTRUCTIONS
    assert "1-2 самых прямых чанка" in _RERANK_INSTRUCTIONS


@pytest.mark.asyncio
async def test_reranker_selects_valid_indices():
    payload = {
        "enough_context": True,
        "selected_indices": [1, 99],
        "reason": "Второй чанк точнее",
    }
    chunks = [
        ContextChunk(text="общий контекст", metadata={"title": "Общее"}),
        ContextChunk(text="точный контекст", metadata={"title": "Дашборд"}),
    ]
    with patch("app.ai.reranker._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=_response(payload))
        with patch("app.ai.reranker.settings.openai_reasoning_effort", "low"):
            result = await rerank_context(
                user_query="вопрос",
                search_query="точный запрос",
                chunks=chunks,
            )

    assert result.enough_context is True
    assert result.selected_indices == [1]
    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["text"]["format"]["type"] == "json_schema"
    assert call_kwargs["reasoning"] == {"effort": "low"}
    assert "previous_response_id" not in call_kwargs


@pytest.mark.asyncio
async def test_reranker_can_reject_context():
    payload = {
        "enough_context": False,
        "selected_indices": [0],
        "reason": "Нет подходящего контекста",
    }
    with patch("app.ai.reranker._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=_response(payload))
        result = await rerank_context(
            user_query="вопрос",
            search_query="точный запрос",
            chunks=[ContextChunk(text="нерелевантно")],
        )

    assert result.enough_context is False
    assert result.selected_indices == []
