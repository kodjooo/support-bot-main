import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from app.config import settings
from app.storage import db
from app.bot.handlers import router

# Таймаут long-polling запроса (секунды)
# Telegram держит соединение не дольше этого значения, затем возвращает пустой ответ
_POLLING_TIMEOUT = 30

# Таймаут HTTP-сессии — должен быть больше polling-таймаута
# aiogram ожидает число (float), а не aiohttp.ClientTimeout
_SESSION_TIMEOUT = _POLLING_TIMEOUT + 15


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    await db.init()

    session = AiohttpSession(timeout=_SESSION_TIMEOUT)
    bot = Bot(token=settings.telegram_bot_token, session=session)
    dp = Dispatcher()
    dp.include_router(router)

    logging.getLogger(__name__).info("Бот запущен")
    await dp.start_polling(bot, timeout=_POLLING_TIMEOUT)


if __name__ == "__main__":
    asyncio.run(main())
