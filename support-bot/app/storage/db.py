import json
import os
import time
from dataclasses import dataclass, field

import aiosqlite

from app.config import settings


@dataclass
class UserRecord:
    user_id: str
    first_name: str
    last_name: str
    last_response_id: str | None  # ID последнего ответа Responses API (для цепочки диалога)
    pending_clarification: dict | None
    texts: list[str]
    image_ids: list[str]
    last_update: int
    # Последние содержательные пары [{"user": ..., "bot": ...}] для разрешения follow-up
    recent_exchanges: list[dict] = field(default_factory=list)


_db_path: str = settings.database_path


async def init() -> None:
    """Создаёт файл БД и таблицу users при первом запуске."""
    os.makedirs(os.path.dirname(_db_path) or ".", exist_ok=True)
    async with aiosqlite.connect(_db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id             TEXT PRIMARY KEY,
                first_name          TEXT,
                last_name           TEXT,
                last_response_id    TEXT,
                pending_clarification_json TEXT,
                texts_json          TEXT,
                image_ids_json      TEXT,
                last_update         INTEGER
            )
        """)
        async with db.execute("PRAGMA table_info(users)") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if "pending_clarification_json" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN pending_clarification_json TEXT")
        if "recent_exchanges_json" not in columns:
            await db.execute("ALTER TABLE users ADD COLUMN recent_exchanges_json TEXT")
        await db.commit()


async def get_user(user_id: str) -> UserRecord | None:
    """Возвращает запись пользователя или None."""
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None

    return UserRecord(
        user_id=row["user_id"],
        first_name=row["first_name"] or "",
        last_name=row["last_name"] or "",
        last_response_id=row["last_response_id"],
        pending_clarification=json.loads(row["pending_clarification_json"] or "null"),
        texts=json.loads(row["texts_json"] or "[]"),
        image_ids=json.loads(row["image_ids_json"] or "[]"),
        last_update=row["last_update"] or 0,
        recent_exchanges=json.loads(row["recent_exchanges_json"] or "[]"),
    )


async def upsert_user(
    user_id: str,
    first_name: str,
    last_name: str,
    texts: list[str],
    image_ids: list[str],
    last_update: int,
) -> None:
    """Создаёт или обновляет запись. last_response_id не трогает."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            INSERT INTO users (user_id, first_name, last_name, texts_json, image_ids_json, last_update)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                first_name      = excluded.first_name,
                last_name       = excluded.last_name,
                texts_json      = excluded.texts_json,
                image_ids_json  = excluded.image_ids_json,
                last_update     = excluded.last_update
            """,
            (
                user_id,
                first_name,
                last_name,
                json.dumps(texts, ensure_ascii=False),
                json.dumps(image_ids, ensure_ascii=False),
                last_update,
            ),
        )
        await db.commit()


async def save_last_response_id(user_id: str, last_response_id: str | None) -> None:
    """Сохраняет ID последнего ответа Responses API для продолжения диалога."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            "UPDATE users SET last_response_id = ? WHERE user_id = ?",
            (last_response_id, user_id),
        )
        await db.commit()


async def append_exchange(user_id: str, user_text: str, bot_answer: str) -> None:
    """Добавляет содержательную пару вопрос-ответ в скользящее окно истории.

    Хранится не более RAG_HISTORY_MAX_PAIRS последних пар; ответ бота
    обрезается до RAG_HISTORY_ANSWER_CHARS символов, чтобы не раздувать
    ввод planner. Самая свежая пара — последняя в списке.
    """
    max_pairs = max(1, settings.rag_history_max_pairs)
    max_chars = max(1, settings.rag_history_answer_chars)
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT recent_exchanges_json FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        history: list[dict] = json.loads(row["recent_exchanges_json"] or "[]")
        history.append({"user": user_text, "bot": (bot_answer or "")[:max_chars]})
        history = history[-max_pairs:]
        await db.execute(
            "UPDATE users SET recent_exchanges_json = ? WHERE user_id = ?",
            (json.dumps(history, ensure_ascii=False), user_id),
        )
        await db.commit()


async def save_pending_clarification(user_id: str, pending: dict) -> None:
    """Сохраняет состояние уточняющего вопроса для следующей реплики пользователя."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            "UPDATE users SET pending_clarification_json = ? WHERE user_id = ?",
            (json.dumps(pending, ensure_ascii=False), user_id),
        )
        await db.commit()


async def clear_pending_clarification(user_id: str) -> None:
    """Очищает состояние уточняющего вопроса."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            "UPDATE users SET pending_clarification_json = NULL WHERE user_id = ?",
            (user_id,),
        )
        await db.commit()


async def clear_buffer(user_id: str) -> None:
    """Очищает буфер текстов и изображений. last_response_id сохраняет."""
    async with aiosqlite.connect(_db_path) as db:
        await db.execute(
            """
            UPDATE users
            SET texts_json = '[]', image_ids_json = '[]', last_update = 0
            WHERE user_id = ?
            """,
            (user_id,),
        )
        await db.commit()


async def consume_buffer(
    user_id: str,
    taken_texts: list[str],
    taken_image_ids: list[str],
) -> None:
    """Удаляет из буфера только те элементы, которые были взяты на обработку.
    Новые сообщения, пришедшие во время обработки, остаются в буфере."""
    async with aiosqlite.connect(_db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT texts_json, image_ids_json FROM users WHERE user_id = ?", (user_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return

        current_texts: list[str] = json.loads(row["texts_json"] or "[]")
        current_images: list[str] = json.loads(row["image_ids_json"] or "[]")

        # Удаляем только обработанные элементы (с начала списка)
        remaining_texts = current_texts[len(taken_texts):]
        remaining_images = current_images[len(taken_image_ids):]

        await db.execute(
            """
            UPDATE users
            SET texts_json = ?, image_ids_json = ?, last_update = CASE WHEN ? > 0 OR ? > 0 THEN last_update ELSE 0 END
            WHERE user_id = ?
            """,
            (
                json.dumps(remaining_texts, ensure_ascii=False),
                json.dumps(remaining_images, ensure_ascii=False),
                len(remaining_texts),
                len(remaining_images),
                user_id,
            ),
        )
        await db.commit()
