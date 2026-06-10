import asyncio

from aiogram import Bot

from app.config import settings

# Словарь активных таймеров: user_id → asyncio.Task
_timers: dict[str, asyncio.Task] = {}


async def debounce(user_id: str, bot: Bot) -> None:
    """Сбрасывает старый таймер и запускает новый для пользователя."""
    if user_id in _timers:
        _timers[user_id].cancel()

    async def _run() -> None:
        await asyncio.sleep(settings.debounce_delay)
        # Убираем себя из словаря — мы вышли из фазы ожидания.
        # Теперь новые сообщения создадут свой таймер, но нас не отменят.
        _timers.pop(user_id, None)
        from app.bot.processor import process_and_reply
        await process_and_reply(bot, user_id)

    _timers[user_id] = asyncio.create_task(_run())
