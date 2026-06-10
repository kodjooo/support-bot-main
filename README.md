# Support Bot Main

Единая рабочая папка для Telegram support-bot, векторной базы знаний и ChromaDB.

Сервисы остаются разделёнными по коду, но запускаются одной командой:

- `support-bot/` — Telegram-бот, OpenAI-ответы, операторские уведомления.
- `vector-base/` — FastAPI `/health` и `/search`, синхронизация Google Docs, RAG-загрузка.
- `chroma` — векторное хранилище.

Архитектура описана в [`docs/architecture.md`](docs/architecture.md). Правила сопровождения лежат в [`docs/arch-rules.md`](docs/arch-rules.md), правила для AI-агента — в [`AGENTS.md`](AGENTS.md).

## Быстрый Старт

```bash
cp .env.example .env
```

Заполните в `.env`:

- `TELEGRAM_BOT_TOKEN` — Telegram `@BotFather`.
- `OPENAI_API_KEY` — https://platform.openai.com/api-keys.
- `OPERATOR_CHAT_ID` — Telegram `@userinfobot`.
- `GOOGLE_DOC_IDS` — ID документов из URL Google Docs.

Положите Google service account key в:

```text
vector-base/secrets/google_service_account.json
```

Запуск:

```bash
make up
```

Проверка:

```bash
make ps
make health
make logs-bot
```

## Основные Команды

```bash
make up           # собрать и поднять весь стек
make down         # остановить стек
make restart      # перезапустить контейнеры
make ps           # показать состояние сервисов
make logs         # все логи
make logs-bot     # логи Telegram-бота
make logs-vector  # логи vector-base
make health       # проверить http://localhost:8080/health
make test         # тесты bot и vector-base
make sync         # ручная полная синхронизация Google Docs
make load-rag     # загрузить подготовленный RAG-корпус в ChromaDB
```

Без `make` можно выполнять те же команды через `docker compose`.

## Связь Сервисов

Бот обращается к векторной базе внутри Docker-сети:

```env
VECTOR_BASE_URL=http://vector-base:8080
```

Локальный IP компьютера для этой связи не используется.

## Ручная Синхронизация

```bash
make sync
```

Или напрямую:

```bash
docker compose run --rm vector-base python -m app.sync_docs --force
```

## Загрузка RAG-Корпуса

```bash
make load-rag
```

Команда загружает `vector-base/artifacts/rag_corpus/rag_chunks.jsonl` в ChromaDB.

## Тесты

```bash
make test
```

Отдельно:

```bash
make test-bot
make test-vector
```

## Документация

- [`docs/architecture.md`](docs/architecture.md) — архитектура стека и сервисов.
- [`docs/arch-rules.md`](docs/arch-rules.md) — правила сопровождения.
- [`AGENTS.md`](AGENTS.md) — правила сопровождения проекта.

Внутренние `docs/`, `README.md`, `AGENTS.md`, `.env.example` и `.gitignore` в подпроектах удалены, а важная информация перенесена в корневую документацию.
`requirements.txt` в подпроектах оставлены намеренно: они нужны Docker-сборке сервисов.

## Диагностика

Если бот не отвечает:

```bash
make ps
make logs-bot
curl -sS "https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/getWebhookInfo"
```

Если бот работает, но отвечает без базы знаний:

```bash
make health
make logs-vector
docker compose exec bot sh -lc 'python - <<PY
from app.config import settings
print(settings.vector_base_url)
PY'
```

Если ChromaDB или vector-base не стартуют, смотрите:

```bash
docker compose logs -f chroma
docker compose logs -f vector-base
```
