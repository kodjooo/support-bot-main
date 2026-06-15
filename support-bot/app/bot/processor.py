import asyncio
import logging

from aiogram import Bot

from app.storage import db
from app.config import settings

logger = logging.getLogger(__name__)

# asyncio.Lock на каждого пользователя
_user_locks: dict[str, asyncio.Lock] = {}


def get_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]


async def _has_new_buffer_items(user_id: str, taken_texts: list[str], taken_image_ids: list[str]) -> bool:
    """Проверяет, пришли ли новые сообщения после взятия снимка буфера."""
    current = await db.get_user(user_id)
    if not current:
        return False
    return len(current.texts) > len(taken_texts) or len(current.image_ids) > len(taken_image_ids)


async def transfer_to_operator(bot: Bot, user_id: str, first_name: str, last_name: str) -> None:
    """Уведомляет пользователя и оператора о переключении на человека."""
    await bot.send_message(
        chat_id=user_id,
        text="Перевожу ваш запрос на оператора. Он подключится в ближайшее время.",
    )
    await bot.send_message(
        chat_id=settings.operator_chat_id,
        text=f"{first_name} {last_name} просит подключиться к его чату!",
    )


async def process_and_reply(bot: Bot, user_id: str) -> None:
    """Основная логика обработки после дебаунса."""
    async with get_lock(user_id):
        import time

        record = await db.get_user(user_id)
        if not record:
            return

        # Если буфер уже пустой (другой вызов успел обработать) — выходим
        if not record.texts and not record.image_ids:
            return

        # TTL-проверка буфера
        if time.time() - record.last_update > settings.max_buffer_age:
            await db.clear_buffer(user_id)
            return

        # Импорт здесь — избегаем циклического импорта на старте
        from app.utils.telegram import get_image_url, keep_typing
        from app.ai.assistant import call_assistant
        from app.ai.cleaner import clean_response
        from app.ai.vector_client import fetch_context
        from app.ai.planner import plan_query
        from app.ai.reranker import rerank_context

        pending = record.pending_clarification or None
        pending_texts = list((pending or {}).get("original_texts") or [])
        pending_image_ids = list((pending or {}).get("original_image_ids") or [])
        effective_texts = pending_texts + record.texts
        effective_image_ids = pending_image_ids + record.image_ids

        # Проверка количества изображений
        if len(effective_image_ids) > settings.max_images:
            await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
            await db.clear_buffer(user_id)
            await db.clear_pending_clarification(user_id)
            return

        # URL изображений — строятся на лету, токен не хранится в БД
        image_urls = []
        for file_id in effective_image_ids:
            url = await get_image_url(bot, file_id)
            image_urls.append(url)

        user_query = "\n".join(effective_texts)
        logger.info("[USER] user_id=%s сообщение: %s", user_id, user_query[:500])

        # Недавний диалог для разрешения follow-up — последние содержательные пары.
        # Передаём только при живой цепочке (last_response_id есть): после перевода
        # на оператора или с нуля цепочка сброшена и история уже неактуальна.
        recent = None
        if record.last_response_id and record.recent_exchanges:
            recent = record.recent_exchanges

        try:
            planning = await asyncio.wait_for(
                plan_query(record.texts, pending=pending, recent=recent, image_urls=image_urls),
                timeout=settings.openai_run_timeout,
            )
        except Exception as e:
            logger.error("Ошибка planner OpenAI (user_id=%s): %s", user_id, e, exc_info=True)
            await db.save_last_response_id(user_id, None)
            await db.clear_pending_clarification(user_id)
            await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
            await db.clear_buffer(user_id)
            return

        if await _has_new_buffer_items(user_id, record.texts, record.image_ids):
            logger.info("Planner обработал устаревший снимок, запускаем повторно (user_id=%s)", user_id)
            from app.bot import debounce
            await debounce.debounce(user_id, bot)
            return

        # Разговорная реплика (приветствие, благодарность, подтверждение):
        # поиск в базе знаний не нужен, отвечает финальный ассистент в контексте диалога.
        # pending не сохраняем и сбрасываем — чтобы реплика не подмешивалась в следующий вопрос.
        # В историю follow-up сохраняем только содержательные (RAG-обоснованные)
        # ответы. Разговорные реплики и уточнения из памяти в окно не попадают,
        # поэтому "ок"/"спасибо" не вытесняют последнюю реальную тему.
        store_history = not planning.is_conversational

        if planning.is_conversational:
            logger.info("Разговорная реплика, пропускаем RAG (user_id=%s)", user_id)
            await db.clear_pending_clarification(user_id)
            # Только новое сообщение пользователя — не подмешиваем незавершённое
            # pending-уточнение, чтобы разговорная реплика не тянула старый вопрос.
            texts = record.texts
        else:
            needs_clarification = (
                not planning.is_ready
                or planning.confidence < settings.rag_planner_confidence_threshold
            )
            if needs_clarification:
                attempts = int((pending or {}).get("attempts") or 0)
                if attempts >= settings.rag_max_clarifications:
                    logger.info("Достигнут лимит уточнений (user_id=%s)", user_id)
                    await db.save_last_response_id(user_id, None)
                    await db.clear_pending_clarification(user_id)
                    await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
                    await db.clear_buffer(user_id)
                    return

                question = (
                    planning.clarifying_question
                    or "Уточните, пожалуйста, о каком разделе, маркетплейсе или показателе идёт речь?"
                )
                await db.save_pending_clarification(
                    user_id,
                    {
                        "original_texts": effective_texts,
                        "original_image_ids": effective_image_ids,
                        "attempts": attempts + 1,
                        "last_question": question,
                        "created_at": int(time.time()),
                    },
                )
                try:
                    await bot.send_message(chat_id=user_id, text=question)
                except Exception as e:
                    logger.error("Ошибка отправки уточняющего вопроса (user_id=%s): %s", user_id, e)
                finally:
                    await db.consume_buffer(user_id, record.texts, record.image_ids)
                return

            await db.clear_pending_clarification(user_id)

            search_query = planning.search_query or user_query
            context_chunks = await fetch_context(search_query, top_k=settings.rag_search_top_k)
            if not context_chunks:
                await db.save_last_response_id(user_id, None)
                await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
                await db.clear_buffer(user_id)
                return

            try:
                rerank = await asyncio.wait_for(
                    rerank_context(
                        user_query=user_query,
                        search_query=search_query,
                        chunks=context_chunks,
                    ),
                    timeout=settings.openai_run_timeout,
                )
            except Exception as e:
                logger.error("Ошибка rerank OpenAI (user_id=%s): %s", user_id, e, exc_info=True)
                await db.save_last_response_id(user_id, None)
                await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
                await db.clear_buffer(user_id)
                return

            if await _has_new_buffer_items(user_id, record.texts, record.image_ids):
                logger.info("Rerank обработал устаревший снимок, запускаем повторно (user_id=%s)", user_id)
                from app.bot import debounce
                await debounce.debounce(user_id, bot)
                return

            selected_chunks = [context_chunks[index] for index in rerank.selected_indices]
            # Контекста нет совсем (reranker не нашёл ни одного тематического чанка) — оператор.
            if not selected_chunks:
                await db.save_last_response_id(user_id, None)
                await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
                await db.clear_buffer(user_id)
                return

            # Мягкий режим: при enough_context=false контекст частичный — передаём его
            # ассистенту с пометкой, а решение "звать оператора" принимает он сам
            # через transfer_to_operator, если фрагментов реально не хватает для ответа.
            if rerank.enough_context:
                context_prefix = (
                    "Контекст из базы знаний уже отобран внутренним RAG pipeline. "
                    "Отвечай только по этим фрагментам. "
                    "Не смешивай фрагменты из разных разделов, если они не отвечают на один и тот же вопрос. "
                    "Не добавляй варианты, шаги или места интерфейса, которых нет в контексте. "
                    "Если контекст не подтверждает ответ, вызови transfer_to_operator.\n"
                )
            else:
                context_prefix = (
                    "Контекст из базы знаний отобран внутренним RAG pipeline, но он частичный — "
                    "прямого исчерпывающего ответа в нём может не быть. "
                    "Отвечай только по этим фрагментам и только в той части, которую они покрывают. "
                    "Не придумывай шаги, варианты или места интерфейса, которых нет в контексте. "
                    "Если фрагменты не дают ответа на вопрос пользователя, вызови transfer_to_operator.\n"
                )
            context_prefix += "\n\n".join(chunk.to_prompt_text() for chunk in selected_chunks)
            texts = [context_prefix] + effective_texts

        # Индикатор "бот печатает..."
        stop_event = asyncio.Event()
        typing_task = asyncio.create_task(keep_typing(bot, user_id, stop_event))

        async def _stop_typing() -> None:
            """Идемпотентно гасит индикатор печати при любом исходе."""
            stop_event.set()
            if not typing_task.done():
                typing_task.cancel()
            await asyncio.gather(typing_task, return_exceptions=True)

        try:
            # Жёсткий таймаут: зависший запрос к OpenAI (например, при битой
            # цепочке previous_response_id, ждущей function_call_output) не должен
            # держать лок пользователя и крутить индикатор печати вечно.
            response_text, needs_operator, new_response_id = await asyncio.wait_for(
                call_assistant(
                    last_response_id=record.last_response_id,
                    texts=texts,
                    image_urls=image_urls,
                ),
                timeout=settings.openai_run_timeout,
            )
        except asyncio.CancelledError:
            # Задача отменена дебаунсом — обязательно останавливаем typing
            await _stop_typing()
            raise
        except Exception as e:
            await _stop_typing()
            if isinstance(e, (asyncio.TimeoutError, TimeoutError)):
                logger.error("OpenAI запрос превысил таймаут %ss (user_id=%s) — перевод на оператора",
                             settings.openai_run_timeout, user_id)
            else:
                logger.error("Ошибка при обращении к OpenAI (user_id=%s): %s", user_id, e, exc_info=True)
            # Сбрасываем last_response_id — он мог остаться в состоянии "ждёт function_call_output"
            await db.save_last_response_id(user_id, None)
            await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
            await db.clear_buffer(user_id)
            return

        await _stop_typing()

        if await _has_new_buffer_items(user_id, record.texts, record.image_ids):
            # Пока OpenAI отвечал, пользователь дописал запрос.
            # Не отправляем частичный ответ и не трогаем буфер:
            # следующая обработка возьмёт старые и новые сообщения одной пачкой.
            logger.info(
                "Ответ на устаревший снимок буфера подавлен, буфер сохранён для повторной обработки "
                "(user_id=%s, taken_texts=%d, taken_images=%d)",
                user_id,
                len(record.texts),
                len(record.image_ids),
            )
            from app.bot import debounce
            await debounce.debounce(user_id, bot)
            return

        if needs_operator:
            # Сбрасываем last_response_id — предыдущий ответ содержал function_call,
            # OpenAI ждёт output для него. Сброс позволяет начать диалог заново.
            await db.save_last_response_id(user_id, None)
            await transfer_to_operator(bot, user_id, record.first_name, record.last_name)
            await db.clear_buffer(user_id)
        else:
            # Сохраняем ID последнего ответа для продолжения диалога
            if new_response_id and new_response_id != record.last_response_id:
                await db.save_last_response_id(user_id, new_response_id)

            cleaned = clean_response(response_text or "")
            try:
                if cleaned:
                    await bot.send_message(chat_id=user_id, text=cleaned)
                    # Сохраняем пару в историю follow-up только для RAG-ответов
                    if store_history:
                        await db.append_exchange(user_id, user_query, cleaned)
                else:
                    logger.warning("Пустой ответ после clean_response (user_id=%s), сообщение не отправлено", user_id)
            except Exception as e:
                logger.error("Ошибка отправки сообщения пользователю (user_id=%s): %s", user_id, e)
            finally:
                # Удаляем из буфера только обработанные сообщения.
                # Новые сообщения, пришедшие пока шёл запрос к OpenAI, остаются.
                await db.consume_buffer(user_id, record.texts, record.image_ids)
                logger.info("consume_buffer выполнен (user_id=%s, taken_texts=%d, taken_images=%d)",
                            user_id, len(record.texts), len(record.image_ids))

        # Проверяем остаток буфера — если пришли новые сообщения во время обработки
        leftover = await db.get_user(user_id)
        logger.info("Остаток буфера после обработки (user_id=%s): texts=%d, images=%d",
                    user_id,
                    len(leftover.texts) if leftover else 0,
                    len(leftover.image_ids) if leftover else 0)

    # После освобождения лока — если остались сообщения, обрабатываем через фоновую задачу
    leftover = await db.get_user(user_id)
    if leftover and (leftover.texts or leftover.image_ids):
        logger.info("Найдены новые сообщения после обработки (user_id=%s) — запускаем повторно", user_id)
        asyncio.create_task(process_and_reply(bot, user_id))
