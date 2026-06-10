import os

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")
os.environ.setdefault("DATABASE_PATH", "/tmp/test.db")

from app.ai.cleaner import clean_response


def test_bold():
    assert clean_response("**жирный текст**") == "жирный текст"


def test_italic():
    assert clean_response("__курсив__") == "курсив"


def test_strikethrough():
    assert clean_response("~~зачёркнутый~~") == "зачёркнутый"


def test_citation_bracket():
    result = clean_response("ответ【1:2†source】здесь")
    assert "†source" not in result
    assert "ответ" in result
    assert "здесь" in result


def test_citation_short():
    result = clean_response("текст【5†source】конец")
    assert "†source" not in result


def test_citation_text_source():
    result = clean_response("смотри [docs†source] подробнее")
    assert "†source" not in result


def test_mixed():
    text = "**Важно:** сделай __это__ и посмотри【1†source】"
    result = clean_response(text)
    assert "**" not in result
    assert "__" not in result
    assert "†source" not in result
    assert "Важно:" in result
    assert "это" in result


def test_no_artifacts():
    text = "Обычный текст без форматирования."
    assert clean_response(text) == text
