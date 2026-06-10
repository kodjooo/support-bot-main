from openai import DefaultHttpxClient, OpenAI

from app.config import Settings


def create_openai_client(settings: Settings, **kwargs) -> OpenAI:
    """Создаёт OpenAI-клиент с опциональным прокси из конфигурации."""
    proxy_url = (settings.openai_proxy_url or "").strip()
    if proxy_url:
        kwargs["http_client"] = DefaultHttpxClient(proxy=_normalize_proxy_url(proxy_url))

    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        **kwargs,
    )


def _normalize_proxy_url(proxy_url: str) -> str:
    """Добавляет схему, если в переменной указан только host:port."""
    if "://" in proxy_url:
        return proxy_url
    return f"http://{proxy_url}"
