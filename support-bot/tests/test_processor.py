import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "9999")
os.environ.setdefault("OPERATOR_NAME", "Оператор")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_processor.db")

import app.storage.db as db
from app.bot.processor import process_and_reply

db._db_path = "/tmp/test_processor.db"


@pytest.fixture(autouse=True)
async def clean_db():
    if os.path.exists("/tmp/test_processor.db"):
        os.remove("/tmp/test_processor.db")
    await db.init()
    yield


async def _seed(user_id: str, texts: list[str], image_ids: list[str], age: int = 0):
    last_update = int(time.time()) - age
    await db.upsert_user(user_id, "Иван", "Петров", texts, image_ids, last_update)


@pytest.mark.asyncio
async def test_sends_reply_on_completed():
    await _seed("1", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_ai.return_value = ("Ответ ассистента", False, "resp_001")
        await process_and_reply(bot, "1")

    bot.send_message.assert_called_once()
    assert bot.send_message.call_args.kwargs["text"] == "Ответ ассистента"
    record = await db.get_user("1")
    assert record.texts == []


@pytest.mark.asyncio
async def test_transfers_when_needs_operator():
    await _seed("2", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_ai.return_value = (None, True, "resp_002")
        await process_and_reply(bot, "2")

    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_skips_stale_buffer():
    await _seed("3", ["старый вопрос"], [], age=7200)
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        await process_and_reply(bot, "3")
        mock_ai.assert_not_called()

    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transfers_on_too_many_images():
    image_ids = [f"img_{i}" for i in range(15)]
    await _seed("4", [], image_ids)
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        await process_and_reply(bot, "4")
        mock_ai.assert_not_called()

    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_saves_new_response_id():
    await _seed("5", ["текст"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_ai.return_value = ("ответ", False, "resp_new_999")
        await process_and_reply(bot, "5")

    record = await db.get_user("5")
    assert record.last_response_id == "resp_new_999"


@pytest.mark.asyncio
async def test_suppresses_stale_reply_when_new_message_arrives_during_processing():
    await _seed("6", ["Меня"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    async def add_message_during_processing(*args, **kwargs):
        await db.upsert_user("6", "Иван", "Петров", ["Меня", "зовут"], [], int(time.time()))
        return "Ответ на старый фрагмент", False, "resp_stale"

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        with patch("app.bot.debounce.debounce", new_callable=AsyncMock) as mock_debounce:
            mock_ai.side_effect = add_message_during_processing
            await process_and_reply(bot, "6")

    bot.send_message.assert_not_called()
    mock_debounce.assert_called_once_with("6", bot)
    record = await db.get_user("6")
    assert record.texts == ["Меня", "зовут"]
    assert record.last_response_id is None
