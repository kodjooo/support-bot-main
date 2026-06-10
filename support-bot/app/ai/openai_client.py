from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from app.config import settings


def create_async_openai_client() -> AsyncOpenAI:
    """Создаёт OpenAI-клиент с опциональным прокси из конфигурации."""
    proxy_url = (settings.openai_proxy_url or "").strip()
    if not proxy_url:
        return AsyncOpenAI(api_key=settings.openai_api_key)

    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        http_client=DefaultAsyncHttpxClient(proxy=_normalize_proxy_url(proxy_url)),
    )


def _normalize_proxy_url(proxy_url: str) -> str:
    """Добавляет схему, если в переменной указан только host:port."""
    if "://" in proxy_url:
        return proxy_url
    return f"http://{proxy_url}"
