import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")
os.environ.setdefault("DATABASE_PATH", "/tmp/test.db")

import app.bot.debounce as deb


@pytest_asyncio.fixture(autouse=True)
async def clear_timers():
    deb._timers.clear()
    yield
    # Отменяем задачи, пока event loop теста ещё активен.
    tasks = list(deb._timers.values())
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    deb._timers.clear()


@pytest.mark.asyncio
async def test_debounce_creates_task():
    bot = AsyncMock()
    with patch("app.config.settings.debounce_delay", 10):
        await deb.debounce("user1", bot)
    assert "user1" in deb._timers
    assert not deb._timers["user1"].done()


@pytest.mark.asyncio
async def test_debounce_cancels_previous():
    bot = AsyncMock()
    with patch("app.config.settings.debounce_delay", 10):
        await deb.debounce("user2", bot)
        first_task = deb._timers["user2"]
        await deb.debounce("user2", bot)
        second_task = deb._timers["user2"]

    # Даём event loop обработать отмену
    await asyncio.sleep(0)
    assert first_task.cancelled()
    assert not second_task.done()


@pytest.mark.asyncio
async def test_debounce_fires_processor():
    bot = AsyncMock()
    process_mock = AsyncMock()

    with patch("app.config.settings.debounce_delay", 0):
        with patch("app.bot.processor.process_and_reply", process_mock):
            await deb.debounce("user3", bot)
            await asyncio.sleep(0.05)

    process_mock.assert_called_once_with(bot, "user3")
