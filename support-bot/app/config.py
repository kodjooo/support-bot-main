from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram
    telegram_bot_token: str
    telegram_proxy_url: str | None = None  # HTTP/SOCKS-прокси для Telegram API; пусто = без прокси

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_planner_model: str | None = None  # модель для внутреннего планирования RAG; пусто = OPENAI_MODEL
    openai_rerank_model: str | None = None  # модель для внутреннего rerank чанков; пусто = OPENAI_MODEL
    openai_system_prompt_file: str = "system_prompt.txt"  # путь к файлу с системным промптом
    openai_temperature: float | None = None  # None = использовать дефолт модели (0.0–2.0); не поддерживается моделями o-серии
    openai_reasoning_effort: str | None = None  # low / medium / high; только для моделей o-серии (o3, o4-mini и др.)
    openai_proxy_url: str | None = None  # HTTP/HTTPS/SOCKS-прокси для всех запросов к OpenAI; пусто = без прокси

    def get_instructions(self) -> str:
        """Читает системный промпт из файла. Возвращает пустую строку если файл не найден."""
        try:
            with open(self.openai_system_prompt_file, encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return ""

    # Оператор
    operator_chat_id: str
    operator_name: str

    # URL сервиса векторного поиска (vector-base)
    # Пример: http://vector-base:8080 (имя сервиса из docker-compose)
    # Если не задан — бот работает без контекста из базы знаний
    vector_base_url: str | None = None

    # База данных
    database_path: str = "./data/chatbot.db"

    # Параметры дебаунса и буфера
    debounce_delay: int = 4
    max_buffer_age: int = 3600
    max_images: int = 10
    min_photo_width: int = 800
    openai_run_timeout: int = 60

    # RAG pipeline
    rag_max_clarifications: int = 2
    rag_search_top_k: int = 6
    rag_rerank_keep: int = 3
    rag_planner_confidence_threshold: float = 0.65


settings = Settings()
