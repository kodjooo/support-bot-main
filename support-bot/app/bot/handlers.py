import time

from aiogram import F, Router
from aiogram.types import Message

from app.config import settings
from app.storage import db
from app.bot import debounce

router = Router()


async def _notify_operator(message: Message) -> None:
    """Отправляет уведомление оператору о пользователе."""
    first = message.from_user.first_name or ""
    last = message.from_user.last_name or ""
    await message.bot.send_message(
        chat_id=settings.operator_chat_id,
        text=f"{first} {last} просит подключиться к его чату!",
    )


async def _save_to_buffer(message: Message, new_text: str = "", new_file_id: str = "") -> None:
    """Дополняет буфер пользователя и запускает дебаунс."""
    user_id = str(message.from_user.id)
    first = message.from_user.first_name or ""
    last = message.from_user.last_name or ""

    record = await db.get_user(user_id)
    texts = (record.texts if record else []) + ([new_text] if new_text else [])
    image_ids = (record.image_ids if record else []) + ([new_file_id] if new_file_id else [])

    await db.upsert_user(
        user_id=user_id,
        first_name=first,
        last_name=last,
        texts=texts,
        image_ids=image_ids,
        last_update=int(time.time()),
    )

    await debounce.debounce(user_id, message.bot)


@router.message(F.chat.type == "private")
async def handle_message(message: Message) -> None:
    """Единый обработчик приватных сообщений: классифицирует и маршрутизирует."""

    # --- Фото ---
    if message.photo:
        largest = message.photo[-1]
        if largest.width < settings.min_photo_width:
            await message.answer(
                "Пожалуйста, загрузите фотографию большего размера — "
                "в текущем виде сложно оценить информацию. "
                "После этого повторите, пожалуйста, свой вопрос."
            )
            await _notify_operator(message)
            return
        await _save_to_buffer(message, new_file_id=largest.file_id)
        return

    # --- Документ-изображение ---
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        await _save_to_buffer(message, new_file_id=message.document.file_id)
        return

    # --- Текст ---
    if message.text:
        await _save_to_buffer(message, new_text=message.text)
        return

    # --- Прочее (видео, аудио, стикеры, Excel, PDF и т.д.) ---
    await message.answer("Перевожу ваш запрос на оператора. Он подключится в ближайшее время.")
    await _notify_operator(message)
