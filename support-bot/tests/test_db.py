import os
import time

import pytest
import pytest_asyncio

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_chatbot.db")

import app.storage.db as db

db._db_path = "/tmp/test_chatbot.db"


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    if os.path.exists("/tmp/test_chatbot.db"):
        os.remove("/tmp/test_chatbot.db")
    await db.init()
    yield


@pytest.mark.asyncio
async def test_get_nonexistent_user():
    result = await db.get_user("999")
    assert result is None


@pytest.mark.asyncio
async def test_upsert_and_get():
    now = int(time.time())
    await db.upsert_user("1", "Иван", "Петров", ["привет"], ["file1"], now)
    record = await db.get_user("1")
    assert record is not None
    assert record.first_name == "Иван"
    assert record.texts == ["привет"]
    assert record.image_ids == ["file1"]
    assert record.last_response_id is None


@pytest.mark.asyncio
async def test_upsert_appends():
    now = int(time.time())
    await db.upsert_user("2", "А", "Б", ["текст1"], [], now)
    rec = await db.get_user("2")
    await db.upsert_user("2", "А", "Б", rec.texts + ["текст2"], rec.image_ids + ["img1"], now)
    rec2 = await db.get_user("2")
    assert rec2.texts == ["текст1", "текст2"]
    assert rec2.image_ids == ["img1"]


@pytest.mark.asyncio
async def test_save_last_response_id():
    now = int(time.time())
    await db.upsert_user("3", "X", "Y", [], [], now)
    await db.save_last_response_id("3", "resp_abc123")
    record = await db.get_user("3")
    assert record.last_response_id == "resp_abc123"


@pytest.mark.asyncio
async def test_clear_buffer_keeps_response_id():
    now = int(time.time())
    await db.upsert_user("4", "A", "B", ["msg"], ["img"], now)
    await db.save_last_response_id("4", "resp_xyz")
    await db.clear_buffer("4")
    record = await db.get_user("4")
    assert record.texts == []
    assert record.image_ids == []
    assert record.last_update == 0
    assert record.last_response_id == "resp_xyz"
