import os
from unittest.mock import MagicMock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("OPENAI_MODEL", "gpt-4o")
os.environ.setdefault("OPERATOR_CHAT_ID", "0")
os.environ.setdefault("OPERATOR_NAME", "test")

from app.ai import openai_client


def test_openai_client_without_proxy_uses_default_network(monkeypatch):
    monkeypatch.setattr(openai_client.settings, "openai_proxy_url", None)

    with patch("app.ai.openai_client.AsyncOpenAI") as mock_openai:
        openai_client.create_async_openai_client()

    assert "http_client" not in mock_openai.call_args.kwargs


def test_openai_client_with_proxy_configures_http_client(monkeypatch):
    monkeypatch.setattr(openai_client.settings, "openai_proxy_url", "127.0.0.1:8080")
    http_client = MagicMock()

    with patch("app.ai.openai_client.DefaultAsyncHttpxClient", return_value=http_client) as mock_http_client:
        with patch("app.ai.openai_client.AsyncOpenAI") as mock_openai:
            openai_client.create_async_openai_client()

    mock_http_client.assert_called_once_with(proxy="http://127.0.0.1:8080")
    assert mock_openai.call_args.kwargs["http_client"] is http_client
