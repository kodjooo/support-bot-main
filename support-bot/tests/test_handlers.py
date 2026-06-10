import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "9999")
os.environ.setdefault("OPERATOR_NAME", "test")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_handlers.db")

import app.storage.db as db
from app.bot.handlers import handle_message

db._db_path = "/tmp/test_handlers.db"


def _make_message(
    text=None, photo=None, document=None, user_id="42", first="Иван", last="Петров"
):
    """Создаёт мок-объект Message."""
    msg = MagicMock()
    msg.from_user.id = int(user_id)
    msg.from_user.first_name = first
    msg.from_user.last_name = last
    msg.text = text
    msg.photo = photo
    msg.document = document
    msg.answer = AsyncMock()
    msg.bot = MagicMock()
    msg.bot.send_message = AsyncMock()
    return msg


@pytest.fixture(autouse=True)
async def clean_db():
    if os.path.exists("/tmp/test_handlers.db"):
        os.remove("/tmp/test_handlers.db")
    await db.init()
    yield


@pytest.mark.asyncio
async def test_text_saved_to_buffer():
    msg = _make_message(text="Привет")
    with patch("app.bot.handlers.debounce.debounce", new_callable=AsyncMock):
        await handle_message(msg)
    record = await db.get_user("42")
    assert record is not None
    assert "Привет" in record.texts


@pytest.mark.asyncio
async def test_small_photo_notifies():
    photo = MagicMock()
    photo.width = 100
    photo.file_id = "small_file"
    msg = _make_message(photo=[photo])
    with patch("app.bot.handlers.debounce.debounce", new_callable=AsyncMock):
        await handle_message(msg)
    msg.answer.assert_called_once()
    msg.bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_normal_photo_saved():
    photo = MagicMock()
    photo.width = 1200
    photo.file_id = "big_file"
    msg = _make_message(photo=[photo])
    with patch("app.bot.handlers.debounce.debounce", new_callable=AsyncMock):
        await handle_message(msg)
    record = await db.get_user("42")
    assert "big_file" in record.image_ids


@pytest.mark.asyncio
async def test_document_image_saved():
    doc = MagicMock()
    doc.mime_type = "image/png"
    doc.file_id = "doc_file"
    msg = _make_message(document=doc)
    with patch("app.bot.handlers.debounce.debounce", new_callable=AsyncMock):
        await handle_message(msg)
    record = await db.get_user("42")
    assert "doc_file" in record.image_ids


@pytest.mark.asyncio
async def test_unsupported_message_notifies():
    msg = _make_message()  # нет ни text, ни photo, ни document
    with patch("app.bot.handlers.debounce.debounce", new_callable=AsyncMock):
        await handle_message(msg)
    msg.answer.assert_called_once()
    msg.bot.send_message.assert_called_once()
