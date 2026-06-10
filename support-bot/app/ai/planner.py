from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from app.ai.openai_client import create_async_openai_client
from app.config import settings

logger = logging.getLogger(__name__)

_client = create_async_openai_client()


@dataclass
class PlanningResult:
    status: str
    clarifying_question: str | None
    search_query: str | None
    extracted: dict
    confidence: float

    @property
    def is_ready(self) -> bool:
        return self.status == "ready" and bool((self.search_query or "").strip())

    @property
    def is_conversational(self) -> bool:
        """Разговорная реплика (приветствие, благодарность, подтверждение):
        planner вернул ready, но поиск в базе знаний не нужен (пустой search_query)."""
        return self.status == "ready" and not (self.search_query or "").strip()


_PLANNER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["ready", "need_clarification"]},
        "clarifying_question": {"type": ["string", "null"]},
        "search_query": {"type": ["string", "null"]},
        "extracted": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "marketplace": {"type": ["string", "null"]},
                "section": {"type": ["string", "null"]},
                "metric": {"type": ["string", "null"]},
                "period": {"type": ["string", "null"]},
                "problem": {"type": ["string", "null"]},
            },
            "required": ["marketplace", "section", "metric", "period", "problem"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["status", "clarifying_question", "search_query", "extracted", "confidence"],
}

_PLANNER_INSTRUCTIONS = """
Ты внутренний planner RAG-бота поддержки Sellerdata. Не отвечай пользователю.
Определи, достаточно ли пользовательского запроса для точного поиска в базе знаний.

Карта базы знаний:
- Дашборд: плитки и таблицы, продажи, возвраты, удержания, реклама, себестоимость, валовая прибыль, расходы, НДС, налог на прибыль, чистая прибыль, сумма выплат, маржа, ROI, выкупаемость, периоды, фильтры, сверка данных Wildberries/Ozon.
- Настройки: API-ключи магазинов, налоги, уведомления, email, пароль, тарифы, оплата, дополнительные услуги, исторические данные, реферальная и партнёрская программы.
- Расходы: таблица расходов, добавление расхода, разовые и повторяющиеся расходы, амортизация, категории, привязка к товарам, почему расход не отображается.
- Самовыкупы: добавление самовыкупа, фильтры, таблица, возвраты, ручные данные, себестоимость, отзыв, детализация, дата выкупа Wildberries, номер заказа.
- Склад: остатки, склады, себестоимость, потенциальные продажи и прибыль, таблица товаров, скорость продаж, остаток в днях, промежуточный склад, закупка, срок заказа, ROI, уведомления об остатках.
- Товары: карточки и фильтры товаров, особенности Wildberries, баркоды, себестоимость, типы себестоимости, даты изменения, загрузка Excel, ошибки загрузки, импорт себестоимости Ozon.

Разговорные реплики (приветствие, благодарность, прощание, короткое подтверждение вроде "спасибо", "ок", "понял", "ясно", "хорошо") не требуют поиска в базе знаний. Для них верни status=ready с пустым search_query (search_query=null) и clarifying_question=null — финальный ассистент ответит сам в контексте диалога. Не задавай уточняющий вопрос на благодарность или подтверждение.

Политика уточнений умеренная:
- Sellerdata — это сервис, а не маркетплейс. Не спрашивай "Wildberries, Ozon или Sellerdata". Если нужно различить, спрашивай отдельно про раздел Sellerdata или про маркетплейс.
- Для терминологических вопросов "что такое X", "что значит X", "где X", "как работает X" обычно верни ready и ищи термин X в базе. Не уточняй до поиска, если термин можно искать сам по себе.
- Если запрос можно сопоставить с одним разделом или одной метрикой из карты базы знаний, верни ready.
- Уточняй только когда без уточнения реально высок риск смешать разные сущности из карты базы знаний.
- Вопросы про расхождение или несовпадение данных ("не сходятся", "не совпадает", "расхождение", "разница", "сверка") зависят от маркетплейса: расхождение всегда означает, что данные в Sellerdata не совпадают с данными маркетплейса, а причины и отчёты для сверки у Wildberries и Ozon разные. Sellerdata — это сервис, а не маркетплейс, поэтому НЕ предлагай "Sellerdata" как сторону расхождения. Если в таком вопросе НЕ назван маркетплейс (нигде нет Wildberries/WB/ВБ/Ozon/Озон), верни need_clarification с коротким вопросом вида "Данные по какому маркетплейсу не сходятся — Wildberries или Ozon?". Не подставляй маркетплейс по умолчанию и не отвечай сразу по обоим. Если маркетплейс в вопросе уже назван (или назван в pending-уточнении) — верни ready и не спрашивай его повторно.
- Реальные оси уточнений: раздел Sellerdata (Дашборд/Настройки/Расходы/Самовыкупы/Склад/Товары), метрика Дашборда (продажи/возвраты/удержания/реклама/себестоимость/прибыль/выплаты/налоги), маркетплейс только если вопрос зависит от Wildberries или Ozon, период только если речь о сверке или расхождении данных.
- Не перечисляй все разделы подряд. Перечисляй только те варианты, где в карте базы реально есть такая сущность.
- Для прибыли реальные варианты: валовая/чистая прибыль в Дашборде или потенциальная прибыль в Складе. Не спрашивай про прибыль в Расходах, Товарах, Настройках или Самовыкупах.
- Если пользователь уже назвал раздел, не спрашивай его повторно. Если уже назвал маркетплейс или понятную аббревиатуру WB/ВБ/Ozon, не спрашивай маркетплейс повторно.
- Для коротких проблем вида "не отображается X" уточняй раздел только если X встречается в нескольких разделах. Если X явно относится к одному разделу карты, верни ready.
- Уточняющий вопрос должен быть один, короткий, на русском, без markdown.
- Уточняющий вопрос должен использовать реальные варианты из карты базы знаний. Не придумывай варианты вроде "резерв денежных средств или резерв товара", если таких сущностей явно нет в карте.
- search_query должен быть обычной естественной поисковой фразой, а не JSON, не key:value и не список через точку с запятой.
- В search_query включай marketplace, раздел, метрику, проблему, период и близкие термины из документации, если они известны.
- Не смешивай разные метрики: продажи, выплаты, реклама, прибыль и налоговая база — это разные темы.
- Если пользователь спрашивает про продажи, не подменяй запрос суммой выплат. Если спрашивает про выплаты, не подменяй продажами или рекламой.
- Для вопроса про расхождение продаж Wildberries с еженедельным отчетом используй термины: сверка данных Wildberries, еженедельные финансовые отчеты, продажи, Дашборд.
- Не придумывай неизвестные значения extracted: ставь null.
"""


async def plan_query(
    user_texts: list[str],
    *,
    pending: dict | None = None,
) -> PlanningResult:
    """Планирует RAG-поиск или уточняющий вопрос. Stateless, без previous_response_id."""
    payload = {
        "current_user_texts": user_texts,
        "pending_clarification": pending,
    }
    params = {
        "model": settings.openai_planner_model or settings.openai_model,
        "instructions": _PLANNER_INSTRUCTIONS.strip(),
        "input": json.dumps(payload, ensure_ascii=False),
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rag_planning_result",
                "schema": _PLANNER_SCHEMA,
                "strict": True,
            },
        },
    }
    if settings.openai_reasoning_effort is not None:
        params["reasoning"] = {"effort": settings.openai_reasoning_effort}
    response = await _client.responses.create(**params)
    data = json.loads(response.output_text or "{}")
    result = PlanningResult(
        status=data["status"],
        clarifying_question=data.get("clarifying_question"),
        search_query=_normalize_search_query(data, user_texts),
        extracted=data.get("extracted") or {},
        confidence=float(data.get("confidence") or 0),
    )
    logger.info(
        "[PLANNER] status=%s confidence=%s search_query=%s question=%s",
        result.status,
        result.confidence,
        (result.search_query or "")[:300],
        result.clarifying_question,
    )
    return result


def _normalize_search_query(data: dict, user_texts: list[str]) -> str | None:
    """Преобразует служебные key:value варианты модели в естественную поисковую фразу."""
    query = (data.get("search_query") or "").strip()
    if not query:
        return None
    if not any(marker in query for marker in (":", ";", "|")):
        return query

    extracted = data.get("extracted") or {}
    parts = [
        str(value).strip()
        for key in ("marketplace", "section", "metric", "problem", "period")
        for value in [extracted.get(key)]
        if value and str(value).strip().lower() != "null"
    ]
    original = " ".join(text.strip() for text in user_texts if text.strip())
    if original:
        parts.append(original)
    if parts:
        return " ".join(dict.fromkeys(parts))
    return query.replace(";", " ").replace("|", " ").replace(":", " ")
