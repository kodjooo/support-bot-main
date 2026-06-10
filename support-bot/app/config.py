from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Telegram
    telegram_bot_token: str

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_system_prompt_file: str = "system_prompt.txt"  # путь к файлу с системным промптом
    openai_temperature: float | None = None  # None = использовать дефолт модели (0.0–2.0); не поддерживается моделями o-серии
    openai_reasoning_effort: str | None = None  # low / medium / high; только для моделей o-серии (o3, o4-mini и др.)

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


settings = Settings()
