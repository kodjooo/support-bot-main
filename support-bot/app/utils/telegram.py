import asyncio

from aiogram import Bot

from app.config import settings


async def get_image_url(bot: Bot, file_id: str) -> str:
    """Строит URL файла на лету. Токен бота нигде не сохраняется."""
    file = await bot.get_file(file_id)
    return f"https://api.telegram.org/file/bot{settings.telegram_bot_token}/{file.file_path}"


async def keep_typing(bot: Bot, chat_id: str, stop_event: asyncio.Event) -> None:
    """Шлёт действие 'typing' каждые 4 секунды пока не установлен stop_event."""
    while not stop_event.is_set():
        await bot.send_chat_action(chat_id=chat_id, action="typing")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=4)
        except asyncio.TimeoutError:
            pass
