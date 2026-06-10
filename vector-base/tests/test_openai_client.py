from unittest.mock import MagicMock, patch

from app.openai_client import create_openai_client


class _Secret:
    def get_secret_value(self) -> str:
        return "key"


class _Settings:
    openai_api_key = _Secret()
    openai_proxy_url = None


def test_openai_client_without_proxy_uses_default_network() -> None:
    settings = _Settings()

    with patch("app.openai_client.OpenAI") as mock_openai:
        create_openai_client(settings)

    assert "http_client" not in mock_openai.call_args.kwargs


def test_openai_client_with_proxy_configures_http_client() -> None:
    settings = _Settings()
    settings.openai_proxy_url = "127.0.0.1:8080"
    http_client = MagicMock()

    with patch("app.openai_client.DefaultHttpxClient", return_value=http_client) as mock_http_client:
        with patch("app.openai_client.OpenAI") as mock_openai:
            create_openai_client(settings, timeout=120.0)

    mock_http_client.assert_called_once_with(proxy="http://127.0.0.1:8080")
    assert mock_openai.call_args.kwargs["http_client"] is http_client
    assert mock_openai.call_args.kwargs["timeout"] == 120.0
