# Support Bot Main

Единая рабочая папка для Telegram support-bot, векторной базы знаний и ChromaDB.

Сервисы остаются разделёнными по коду, но запускаются одной командой:

- `support-bot/` — Telegram-бот, OpenAI-ответы, операторские уведомления.
- `vector-base/` — FastAPI `/health` и `/search`, синхронизация Google Docs, RAG-загрузка.
- `chroma` — векторное хранилище.

Архитектура описана в [`docs/architecture.md`](docs/architecture.md). Правила сопровождения лежат в [`docs/arch-rules.md`](docs/arch-rules.md).

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
make health       # проверить опубликованный /health vector-base
make test         # тесты bot и vector-base
make sync         # ручная полная синхронизация Google Docs
make load-rag     # загрузить подготовленный RAG-корпус в ChromaDB
```

Без `make` можно выполнять те же команды через `docker compose`.

## Связь Сервисов

Бот обращается к векторной базе внутри Docker-сети:

```env
VECTOR_BASE_URL=http://vector-base:8080
VECTOR_BASE_PUBLISHED_PORT=8081
```

`VECTOR_BASE_URL` используется только внутри Docker-сети. `VECTOR_BASE_PUBLISHED_PORT` задаёт внешний порт на хосте для ручной проверки `/health` и `/search`. Если порт `8080` нужен локальному OpenAI-прокси, оставьте `VECTOR_BASE_PUBLISHED_PORT=8081`.

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

## Развёртывание На Сервере

Требования на сервере:

- Linux-сервер с Docker и Docker Compose v2.
- Доступ к репозиторию `https://github.com/kodjooo/support-bot-main.git`.
- Telegram bot token, OpenAI API key, Google service account key.

Установка Docker на Ubuntu/Debian:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER"
```

После добавления пользователя в группу `docker` перелогиньтесь на сервер.

Клонирование и настройка:

```bash
git clone https://github.com/kodjooo/support-bot-main.git
cd support-bot-main
cp .env.example .env
nano .env
```

Заполните `.env` реальными значениями. Google service account key положите в:

```text
vector-base/secrets/google_service_account.json
```

Если OpenAI должен ходить через локальный прокси на Docker-хосте, укажите:

```env
OPENAI_PROXY_URL=socks5://host.docker.internal:8080
```

Если Telegram API тоже недоступен напрямую с сервера, укажите:

```env
TELEGRAM_PROXY_URL=socks5://host.docker.internal:8080
```

`127.0.0.1:8080` внутри контейнера указывает на сам контейнер. Для прокси, запущенного на сервере или локальном компьютере рядом с Docker, используйте `host.docker.internal`; в `docker-compose.yml` добавлен `host-gateway` для Linux. Для SSH dynamic forwarding (`ssh -D 127.0.0.1:8080 ...`) используйте схему `socks5://`.

Если Linux-серверный SOCKS-прокси слушает только `127.0.0.1`, контейнеры не смогут подключиться к нему напрямую через `host.docker.internal`. Включите compose-профиль forwarder:

```env
COMPOSE_PROFILES=host-proxy-forwarder
HOST_PROXY_SOURCE_PORT=8080
HOST_PROXY_FORWARD_PORT=18080
OPENAI_PROXY_URL=socks5://host.docker.internal:18080
TELEGRAM_PROXY_URL=socks5://host.docker.internal:18080
```

Запуск:

```bash
docker compose up -d --build
```

Проверка:

```bash
docker compose ps
make health
docker compose logs -f bot
```

Загрузка подготовленного RAG-корпуса:

```bash
docker compose exec vector-base python -m app.load_rag_corpus artifacts/rag_corpus/rag_chunks.jsonl
```

Обновление версии на сервере:

```bash
git pull
docker compose up -d --build
```

Остановка:

```bash
docker compose down
```

## Документация

- [`docs/architecture.md`](docs/architecture.md) — архитектура стека и сервисов.
- [`docs/arch-rules.md`](docs/arch-rules.md) — правила сопровождения.

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
