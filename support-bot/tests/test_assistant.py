import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")
os.environ.setdefault("DATABASE_PATH", "/tmp/test.db")

from app.ai.assistant import call_assistant


def _make_response(status: str, output_text: str = "", tool_name: str | None = None):
    """Создаёт мок-объект Response API."""
    resp = MagicMock()
    resp.id = "resp_test123"
    resp.status = status
    resp.output_text = output_text

    output_items = []
    if tool_name:
        item = MagicMock()
        item.type = "function_tool_call"
        item.name = tool_name
        output_items.append(item)
    else:
        item = MagicMock()
        item.type = "message"
        output_items.append(item)

    resp.output = output_items
    return resp


@pytest.mark.asyncio
async def test_completed_returns_text():
    resp = _make_response("completed", output_text="Привет!")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        text, needs_op, rid = await call_assistant(None, ["вопрос"], [])

    assert text == "Привет!"
    assert needs_op is False
    assert rid == "resp_test123"


@pytest.mark.asyncio
async def test_transfer_to_operator_via_function_call():
    resp = _make_response("completed", tool_name="transfer_to_operator")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        text, needs_op, rid = await call_assistant(None, ["вопрос"], [])

    assert needs_op is True
    assert text is None


@pytest.mark.asyncio
async def test_failed_status_transfers():
    resp = _make_response("failed")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        text, needs_op, rid = await call_assistant(None, [], [])

    assert needs_op is True


@pytest.mark.asyncio
async def test_previous_response_id_passed():
    resp = _make_response("completed", output_text="ок")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        await call_assistant("resp_prev_abc", ["текст"], [])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert call_kwargs["previous_response_id"] == "resp_prev_abc"


@pytest.mark.asyncio
async def test_no_previous_id_on_first_message():
    resp = _make_response("completed", output_text="ок")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        await call_assistant(None, ["текст"], [])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    assert "previous_response_id" not in call_kwargs


@pytest.mark.asyncio
async def test_image_urls_in_input():
    resp = _make_response("completed", output_text="вижу")
    with patch("app.ai.assistant._client") as mock_client:
        mock_client.responses.create = AsyncMock(return_value=resp)
        await call_assistant(None, ["текст"], ["https://example.com/img.jpg"])

    call_kwargs = mock_client.responses.create.call_args.kwargs
    content = call_kwargs["input"][0]["content"]
    types = [item["type"] for item in content]
    assert "input_image" in types
