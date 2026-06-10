import logging

from openai import AsyncOpenAI, BadRequestError

from app.config import settings

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=settings.openai_api_key)

# Инструмент Function Calling для перевода на оператора
OPERATOR_TOOLS = [
    {
        "type": "function",
        "name": "transfer_to_operator",
        "description": (
            "Вызывай эту функцию в следующих случаях:\n"
            "1. Пользователь явно просит соединить его с живым человеком, оператором или менеджером.\n"
            "2. Пользователь явно недоволен тем, как ты отвечаешь.\n"
            "3. Ответа на вопрос нет в базе знаний или информация неоднозначна.\n"
            "4. Информации недостаточно или ты не уверен в правильности ответа."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
        "strict": False,
    }
]


async def call_assistant(
    last_response_id: str | None,
    texts: list[str],
    image_urls: list[str],
) -> tuple[str | None, bool, str | None]:
    """
    Отправляет сообщение через OpenAI Responses API и возвращает:
      (response_text, needs_operator, new_last_response_id)

    Если needs_operator=True — response_text будет None.
    Бросает исключение при любой сетевой или API ошибке.
    """
    combined_text = "\n".join(texts) if texts else ""

    # Формируем input.
    # Responses API требует обёртки в {"role": "user", "content": [...]} когда есть картинки.
    # Без картинок — простая строка.
    if image_urls:
        content: list[dict] = []
        if combined_text:
            content.append({"type": "input_text", "text": combined_text})
        for url in image_urls:
            content.append({"type": "input_image", "image_url": url})
        input_data = [{"role": "user", "content": content}]
    else:
        input_data = combined_text

    # Параметры запроса
    params: dict = {
        "model": settings.openai_model,
        "input": input_data,
        "tools": OPERATOR_TOOLS,
    }

    # Системный промпт — только при первом обращении (нет previous_response_id)
    if last_response_id is None:
        instructions = settings.get_instructions()
        if instructions:
            params["instructions"] = instructions

    # Продолжение существующего диалога
    if last_response_id is not None:
        params["previous_response_id"] = last_response_id

    # Температура — для gpt-серии (0.0–2.0); модели o-серии её не поддерживают
    if settings.openai_temperature is not None:
        params["temperature"] = settings.openai_temperature

    # Уровень рассуждения — только для моделей o-серии (o3, o4-mini и др.)
    if settings.openai_reasoning_effort is not None:
        params["reasoning"] = {"effort": settings.openai_reasoning_effort}

    logger.info("[OPENAI] Запрос: model=%s, last_response_id=%s, texts_count=%d, images_count=%d",
                settings.openai_model, last_response_id, len(texts), len(image_urls))
    logger.info("[OPENAI] Полный текст запроса: %s", combined_text[:1000])

    try:
        response = await _client.responses.create(**params)
    except BadRequestError as e:
        logger.error("OpenAI 400 Bad Request: %s", e.message)
        raise
    except Exception as e:
        logger.error("OpenAI API ошибка: %s", e)
        raise

    new_last_response_id = response.id
    logger.info("OpenAI response: id=%s, status=%s, output_items=%d",
                response.id, response.status, len(response.output))

    # Проверяем, вызвал ли ассистент transfer_to_operator
    # Тип может быть "function_call" или "function_tool_call" в зависимости от версии API
    for item in response.output:
        item_type = getattr(item, "type", None)
        item_name = getattr(item, "name", None)
        logger.info("OpenAI output item: type=%s, name=%s", item_type, item_name)
        if item_type in ("function_call", "function_tool_call"):
            if item_name == "transfer_to_operator":
                logger.info("OpenAI вызвал transfer_to_operator — переводим на оператора")
                return None, True, new_last_response_id

    # Получаем текстовый ответ
    if response.status == "completed":
        output_text = response.output_text
        logger.info("[OPENAI] Полный ответ: %s", output_text or "")
        return output_text, False, new_last_response_id

    # Все остальные статусы (failed, incomplete и т.д.)
    logger.warning("OpenAI response status: %s", response.status)
    return None, True, new_last_response_id
