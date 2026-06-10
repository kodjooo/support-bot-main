import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "9999")
os.environ.setdefault("OPERATOR_NAME", "Оператор")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_processor.db")

import app.storage.db as db
from app.ai.planner import PlanningResult
from app.ai.reranker import RerankResult
from app.ai.vector_client import ContextChunk
from app.bot.processor import process_and_reply

db._db_path = "/tmp/test_processor.db"


@pytest_asyncio.fixture(autouse=True)
async def clean_db():
    if os.path.exists("/tmp/test_processor.db"):
        os.remove("/tmp/test_processor.db")
    await db.init()
    yield


async def _seed(user_id: str, texts: list[str], image_ids: list[str], age: int = 0):
    last_update = int(time.time()) - age
    await db.upsert_user(user_id, "Иван", "Петров", texts, image_ids, last_update)


def _ready_plan(search_query: str = "точный запрос") -> PlanningResult:
    return PlanningResult(
        status="ready",
        clarifying_question=None,
        search_query=search_query,
        extracted={"marketplace": None, "section": None, "metric": None, "period": None, "problem": None},
        confidence=0.9,
    )


def _clarify_plan(question: str = "Уточните раздел?") -> PlanningResult:
    return PlanningResult(
        status="need_clarification",
        clarifying_question=question,
        search_query=None,
        extracted={"marketplace": None, "section": None, "metric": None, "period": None, "problem": None},
        confidence=0.4,
    )


def _chunk(text: str = "Контекст") -> ContextChunk:
    return ContextChunk(text=text, score=1.0, metadata={"title": "Дашборд"})


def _rerank(indices: list[int] | None = None, enough: bool = True) -> RerankResult:
    return RerankResult(enough_context=enough, selected_indices=indices or [0], reason="ok")


@pytest.mark.asyncio
async def test_sends_reply_on_completed():
    await _seed("1", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("Ответ ассистента", False, "resp_001")
        await process_and_reply(bot, "1")

    bot.send_message.assert_called_once()
    assert bot.send_message.call_args.kwargs["text"] == "Ответ ассистента"
    record = await db.get_user("1")
    assert record.texts == []


@pytest.mark.asyncio
async def test_transfers_when_needs_operator():
    await _seed("2", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = (None, True, "resp_002")
        await process_and_reply(bot, "2")

    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_skips_stale_buffer():
    await _seed("3", ["старый вопрос"], [], age=7200)
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        await process_and_reply(bot, "3")
        mock_ai.assert_not_called()

    bot.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_transfers_on_too_many_images():
    image_ids = [f"img_{i}" for i in range(15)]
    await _seed("4", [], image_ids)
    bot = MagicMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        await process_and_reply(bot, "4")
        mock_ai.assert_not_called()

    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_saves_new_response_id():
    await _seed("5", ["текст"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("ответ", False, "resp_new_999")
        await process_and_reply(bot, "5")

    record = await db.get_user("5")
    assert record.last_response_id == "resp_new_999"


@pytest.mark.asyncio
async def test_typing_error_does_not_block_reply():
    await _seed("7", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock(side_effect=RuntimeError("telegram timeout"))
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("Ответ после сбоя typing", False, "resp_typing_error")
        await process_and_reply(bot, "7")

    bot.send_message.assert_called_once()
    assert bot.send_message.call_args.kwargs["text"] == "Ответ после сбоя typing"
    record = await db.get_user("7")
    assert record.texts == []


@pytest.mark.asyncio
async def test_typing_uses_short_telegram_timeout():
    await _seed("8", ["вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    async def delayed_reply(*args, **kwargs):
        await asyncio.sleep(0)
        return "Ответ", False, "resp_typing_timeout"

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.side_effect = delayed_reply
        await process_and_reply(bot, "8")

    bot.send_chat_action.assert_called()
    assert bot.send_chat_action.call_args.kwargs["request_timeout"] == 3
    bot.send_message.assert_called_once()


@pytest.mark.asyncio
async def test_suppresses_stale_reply_when_new_message_arrives_during_processing():
    await _seed("6", ["Меня"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    async def add_message_during_processing(*args, **kwargs):
        await db.upsert_user("6", "Иван", "Петров", ["Меня", "зовут"], [], int(time.time()))
        return "Ответ на старый фрагмент", False, "resp_stale"

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        with patch("app.bot.debounce.debounce", new_callable=AsyncMock) as mock_debounce:
            mock_plan.return_value = _ready_plan()
            mock_vector.return_value = [_chunk()]
            mock_rerank.return_value = _rerank()
            mock_ai.side_effect = add_message_during_processing
            await process_and_reply(bot, "6")

    bot.send_message.assert_not_called()
    mock_debounce.assert_called_once_with("6", bot)
    record = await db.get_user("6")
    assert record.texts == ["Меня", "зовут"]
    assert record.last_response_id is None


@pytest.mark.asyncio
async def test_asks_clarification_and_skips_vector_search():
    await _seed("9", ["у меня не сходятся продажи"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _clarify_plan("Где именно не сходятся продажи?")
        await process_and_reply(bot, "9")

    bot.send_message.assert_called_once_with(chat_id="9", text="Где именно не сходятся продажи?")
    mock_vector.assert_not_called()
    mock_ai.assert_not_called()
    record = await db.get_user("9")
    assert record.texts == []
    assert record.pending_clarification["attempts"] == 1
    assert record.pending_clarification["original_texts"] == ["у меня не сходятся продажи"]


@pytest.mark.asyncio
async def test_pending_answer_is_combined_and_search_uses_planned_query():
    await _seed("10", ["в дашборде WB"], [])
    await db.save_pending_clarification(
        "10",
        {
            "original_texts": ["у меня не сходятся продажи"],
            "original_image_ids": [],
            "attempts": 1,
            "last_question": "Где именно?",
            "created_at": int(time.time()),
        },
    )
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan("расхождение продаж WB в Дашборде")
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("ответ", False, "resp_10")
        await process_and_reply(bot, "10")

    mock_plan.assert_called_once()
    assert mock_plan.call_args.kwargs["pending"]["attempts"] == 1
    mock_vector.assert_called_once_with("расхождение продаж WB в Дашборде", top_k=10)
    assistant_texts = mock_ai.call_args.kwargs["texts"]
    assert "у меня не сходятся продажи" in assistant_texts[-2]
    assert "в дашборде WB" in assistant_texts[-1]
    record = await db.get_user("10")
    assert record.pending_clarification is None
    assert record.texts == []


@pytest.mark.asyncio
async def test_transfers_after_clarification_limit():
    await _seed("11", ["не знаю"], [])
    await db.save_pending_clarification(
        "11",
        {
            "original_texts": ["у меня не сходятся продажи"],
            "original_image_ids": [],
            "attempts": 2,
            "last_question": "Где именно?",
            "created_at": int(time.time()),
        },
    )
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan:
        mock_plan.return_value = _clarify_plan()
        await process_and_reply(bot, "11")

    assert bot.send_message.call_count == 2
    record = await db.get_user("11")
    assert record.texts == []
    assert record.pending_clarification is None


@pytest.mark.asyncio
async def test_reranker_selected_chunks_are_sent_to_assistant():
    await _seed("12", ["конкретный вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()
    chunks = [_chunk("лишний контекст"), _chunk("нужный контекст")]

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan("точный запрос")
        mock_vector.return_value = chunks
        mock_rerank.return_value = _rerank([1])
        mock_ai.return_value = ("ответ", False, "resp_12")
        await process_and_reply(bot, "12")

    context = mock_ai.call_args.kwargs["texts"][0]
    assert "нужный контекст" in context
    assert "лишний контекст" not in context


@pytest.mark.asyncio
async def test_transfers_when_reranker_rejects_context():
    """Reranker не нашёл ни одного тематического чанка (selected пустой) → оператор."""
    await _seed("13", ["конкретный вопрос"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = RerankResult(enough_context=False, selected_indices=[], reason="нет темы")
        await process_and_reply(bot, "13")

    mock_ai.assert_not_called()
    assert bot.send_message.call_count == 2


@pytest.mark.asyncio
async def test_partial_context_is_passed_to_assistant():
    """Мягкий режим: enough_context=false, но есть чанки → ассистент получает частичный контекст."""
    await _seed("14", ["как отменить подписку"], [])
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan("отмена подписки Настройки тарифы")
        mock_vector.return_value = [_chunk("про оплату и тарифы")]
        mock_rerank.return_value = _rerank([0], enough=False)
        mock_ai.return_value = ("частичный ответ", False, "resp_14")
        await process_and_reply(bot, "14")

    mock_ai.assert_called_once()
    context = mock_ai.call_args.kwargs["texts"][0]
    assert "частичный" in context
    assert "про оплату и тарифы" in context
    bot.send_message.assert_called_once_with(chat_id="14", text="частичный ответ")


@pytest.mark.asyncio
async def test_recent_dialogue_passed_to_planner_and_saved():
    """Follow-up: при живой цепочке planner получает recent_dialogue, а после ответа пара сохраняется."""
    await _seed("16", ["а заказы в пути там учитываются?"], [])
    await db.save_last_response_id("16", "resp_prev")
    await db.save_last_exchange("16", "как считается выкупаемость?", "Выкупаемость = выкупленные/завершённые...")
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan("выкупаемость заказы в пути учитываются")
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("В выкупаемости заказы в пути не учитываются.", False, "resp_16")
        await process_and_reply(bot, "16")

    recent = mock_plan.call_args.kwargs["recent"]
    assert recent == {
        "last_user_text": "как считается выкупаемость?",
        "last_bot_answer": "Выкупаемость = выкупленные/завершённые...",
    }
    record = await db.get_user("16")
    assert record.last_user_text == "а заказы в пути там учитываются?"
    assert record.last_bot_answer == "В выкупаемости заказы в пути не учитываются."


@pytest.mark.asyncio
async def test_recent_dialogue_not_passed_without_active_chain():
    """Без last_response_id прошлая пара не передаётся (цепочка сброшена)."""
    await _seed("17", ["новый вопрос"], [])
    await db.save_last_exchange("17", "старый вопрос", "старый ответ")
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = _ready_plan()
        mock_vector.return_value = [_chunk()]
        mock_rerank.return_value = _rerank()
        mock_ai.return_value = ("ответ", False, "resp_17")
        await process_and_reply(bot, "17")

    assert mock_plan.call_args.kwargs["recent"] is None


@pytest.mark.asyncio
async def test_conversational_reply_skips_rag():
    """Разговорная реплика идёт прямо к ассистенту без поиска и без контекста."""
    await _seed("15", ["спасибо"], [])
    await db.save_last_response_id("15", "resp_prev")
    await db.save_pending_clarification(
        "15",
        {
            "original_texts": ["прошлый вопрос"],
            "original_image_ids": [],
            "attempts": 1,
            "last_question": "Где?",
            "created_at": int(time.time()),
        },
    )
    bot = MagicMock()
    bot.get_file = AsyncMock()
    bot.send_chat_action = AsyncMock()
    bot.send_message = AsyncMock()

    conversational = PlanningResult(
        status="ready",
        clarifying_question=None,
        search_query=None,
        extracted={"marketplace": None, "section": None, "metric": None, "period": None, "problem": None},
        confidence=0.9,
    )

    with patch("app.ai.planner.plan_query", new_callable=AsyncMock) as mock_plan, \
            patch("app.ai.vector_client.fetch_context", new_callable=AsyncMock) as mock_vector, \
            patch("app.ai.reranker.rerank_context", new_callable=AsyncMock) as mock_rerank, \
            patch("app.ai.assistant.call_assistant", new_callable=AsyncMock) as mock_ai:
        mock_plan.return_value = conversational
        mock_ai.return_value = ("Пожалуйста!", False, "resp_15")
        await process_and_reply(bot, "15")

    mock_vector.assert_not_called()
    mock_rerank.assert_not_called()
    mock_ai.assert_called_once()
    # Ассистент получает только текст пользователя без RAG-префикса
    assert mock_ai.call_args.kwargs["texts"] == ["спасибо"]
    bot.send_message.assert_called_once_with(chat_id="15", text="Пожалуйста!")
    record = await db.get_user("15")
    assert record.pending_clarification is None
