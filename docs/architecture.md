# Архитектура support-bot-main

`support-bot-main` — единый Docker Compose проект для Telegram-бота поддержки, векторной базы знаний и ChromaDB. Раньше бот и vector-base жили как отдельные проекты; теперь это один репозиторий с двумя сервисными подпапками.

Кратко:

- `bot` получает сообщения Telegram, запрашивает контекст в `vector-base` и отвечает через OpenAI Responses API.
- `vector-base` синхронизирует Google Docs, хранит embeddings в ChromaDB и отдаёт контекст через `/search`.
- `chroma` доступна только внутри Docker-сети.

## Цели Архитектуры

- Один репозиторий и один корневой `docker-compose.yml`.
- Один корневой `.env` для всего runtime.
- Один набор документации в корневой папке `docs/`.
- Разделение кода по ответственности: Telegram-бот отдельно, векторная база отдельно.
- Связь сервисов через Docker DNS, без локальных IP адресов.

## Сервисы

### `bot`

Путь к коду: `support-bot/`.

Отвечает за:

- получение Telegram updates через long polling;
- накопление сообщений пользователя в SQLite;
- debounce-паузу перед обработкой;
- получение контекста из `vector-base`;
- вызов OpenAI Responses API;
- отправку ответа пользователю;
- перевод на оператора при запросе, ошибке, таймауте или недостаточном контексте.

Основные модули:

- `support-bot/app/main.py` — старт aiogram polling.
- `support-bot/app/config.py` — настройки из корневого `.env`.
- `support-bot/app/bot/handlers.py` — классификация входящих сообщений.
- `support-bot/app/bot/debounce.py` — таймер ожидания пользовательской паузы.
- `support-bot/app/bot/processor.py` — основной сценарий обработки.
- `support-bot/app/ai/vector_client.py` — HTTP-клиент к `vector-base`.
- `support-bot/app/ai/assistant.py` — OpenAI Responses API и function calling `transfer_to_operator`.
- `support-bot/app/ai/cleaner.py` — очистка ответа от markdown/citation-артефактов.
- `support-bot/app/storage/db.py` — SQLite-буфер пользователя и `last_response_id`.
- `support-bot/app/utils/telegram.py` — URL изображений и индикатор "печатает".

### Поведение `bot`

Бот обрабатывает только личные сообщения Telegram. Группы и каналы не являются целевым сценарием.

Классификация входящих сообщений:

- текст — добавляется в буфер `texts`;
- фото шириной не меньше `MIN_PHOTO_WIDTH` — сохраняется `file_id` наибольшего фото;
- фото меньше `MIN_PHOTO_WIDTH` — пользователь получает сообщение, оператор получает уведомление, файл не сохраняется;
- документ с `mime_type`, начинающимся на `image/` — сохраняется `file_id`;
- видео, аудио и прочие неподдерживаемые типы — пользователь получает сообщение, оператор получает уведомление, данные не сохраняются.

В буфере хранятся только Telegram `file_id`, а не URL файлов. URL с токеном бота строится на лету перед отправкой в OpenAI и не сохраняется в SQLite.

После каждого сообщения запускается debounce-таймер. Новое сообщение того же пользователя отменяет старый таймер и запускает новый. Альбомы Telegram обрабатываются этой же механикой: отдельные updates накапливаются и уходят в обработку одной пачкой после паузы.

Перед обработкой буфера выполняются проверки:

- буфер старше `MAX_BUFFER_AGE` очищается без ответа;
- если изображений больше `MAX_IMAGES`, запрос переводится оператору;
- параллельная обработка одного пользователя блокируется через per-user `asyncio.Lock`.

Пока OpenAI обрабатывает запрос, бот отправляет Telegram chat action `typing`. Ошибка отправки этого индикатора логируется как предупреждение и не прерывает основной ответ пользователю. Если пользователь написал новое сообщение во время OpenAI-запроса, ответ на старый снимок буфера подавляется, буфер не очищается, а обработка запускается повторно уже с полной актуальной пачкой.

OpenAI вызывается через Responses API. Для продолжения диалога используется `last_response_id`, сохранённый в SQLite. При function call `transfer_to_operator`, ошибке API или таймауте `OPENAI_RUN_TIMEOUT` бот сбрасывает `last_response_id`, уведомляет пользователя и отправляет сообщение оператору. Рабочий таймаут ожидания OpenAI задан в конфигурации как 300 секунд, чтобы длинные ответы модели не переводились оператору слишком рано.

Все OpenAI-запросы бота создаются через `support-bot/app/ai/openai_client.py`. Если в `.env` задан `OPENAI_PROXY_URL`, Responses API использует этот HTTP/HTTPS/SOCKS5-прокси. Если переменная пустая, клиент работает через обычную сеть.

Telegram long polling создаётся в `support-bot/app/main.py`. Если задан `TELEGRAM_PROXY_URL`, aiogram использует этот HTTP/HTTPS/SOCKS5-прокси для Telegram API. Это нужно на серверах, где `api.telegram.org` недоступен напрямую.

Для Linux-серверов с SSH SOCKS-прокси, который слушает только `127.0.0.1`, в `docker-compose.yml` есть профиль `host-proxy-forwarder`. Он запускает `alpine/socat` в host network и пробрасывает `127.0.0.1:${HOST_PROXY_SOURCE_PORT:-8080}` на `0.0.0.0:${HOST_PROXY_FORWARD_PORT:-18080}`, после чего контейнеры используют `socks5://host.docker.internal:${HOST_PROXY_FORWARD_PORT}`.

### `vector-base`

Путь к коду: `vector-base/`.

Отвечает за:

- чтение Google Docs через service account;
- проверку `modifiedTime` документов через Google Drive API;
- хранение состояния синхронизации в `vector-base/meta/`;
- разбиение текста на чанки;
- генерацию embeddings через OpenAI Embeddings API;
- хранение embeddings и документов в ChromaDB;
- HTTP API `/health` и `/search`;
- загрузку подготовленного RAG-корпуса из `vector-base/artifacts/rag_corpus/rag_chunks.jsonl`;
- плановую синхронизацию через APScheduler.

Основные модули:

- `vector-base/app/main.py` — старт FastAPI и планировщика.
- `vector-base/app/api.py` — HTTP API поиска.
- `vector-base/app/config.py` — настройки из корневого `.env`.
- `vector-base/app/google_docs.py` — Google Docs/Drive API.
- `vector-base/app/embeddings.py` — чанкинг и OpenAI embeddings.
- `vector-base/app/chroma_manager.py` — ChromaDB, поиск, keyword-кандидаты, rerank.
- `vector-base/app/sync_docs.py` — CLI и orchestrator синхронизации Google Docs.
- `vector-base/app/load_rag_corpus.py` — загрузка JSONL RAG-корпуса.

Все OpenAI-запросы `vector-base`, включая embeddings и генератор RAG-корпуса, создаются через `vector-base/app/openai_client.py` и используют `OPENAI_PROXY_URL` при наличии значения.

### `chroma`

ChromaDB хранит embeddings и документы. Сервис доступен только внутри Docker-сети как `chroma:8000`; наружу порт ChromaDB не публикуется.

## Поток Сообщения

1. Пользователь отправляет сообщение в Telegram.
2. `bot` сохраняет текст и изображения в SQLite.
3. `debounce.py` ждёт паузу `DEBOUNCE_DELAY`.
4. `processor.py` берёт снимок буфера под per-user lock.
5. `vector_client.py` отправляет запрос в `http://vector-base:8080/search`.
6. `vector-base` генерирует embedding запроса, ищет semantic-кандидатов в ChromaDB, добавляет keyword-кандидатов и выполняет rerank.
7. `bot` добавляет найденный контекст к пользовательскому запросу.
8. `assistant.py` вызывает OpenAI Responses API.
9. `bot` отправляет очищенный ответ пользователю или переводит запрос оператору.
10. Обработанные элементы удаляются из буфера. Новые сообщения, пришедшие во время ответа, остаются и обрабатываются следующим проходом.

## Vector Search API

`POST /search` принимает:

```json
{"query": "текст вопроса", "top_k": 3}
```

Ответ содержит:

- `chunks` — список текстовых фрагментов для обратной совместимости.
- `results` — подробные результаты с `metadata`, `distance`, `score`, `semantic_score`, `keyword_score`, `matched_terms`.

Параметры качества поиска:

- `SEARCH_TOP_K` — сколько результатов вернуть.
- `SEARCH_CANDIDATE_MULTIPLIER` — сколько semantic-кандидатов взять до rerank.
- `SEARCH_MIN_SCORE` — минимальный итоговый score.
- `SEARCH_KEYWORD_LIMIT` — глубина keyword-просмотра документов коллекции.

## Docker И Сеть

Все сервисы находятся в сети `support-net`.

```env
VECTOR_BASE_URL=http://vector-base:8080
CHROMA_HOST=chroma
CHROMA_PORT=8000
```

Наружу публикуется только HTTP API `vector-base`; порт на хосте задаётся `VECTOR_BASE_PUBLISHED_PORT`:

```text
localhost:${VECTOR_BASE_PUBLISHED_PORT:-8080} -> vector-base:8080
```

Для локального OpenAI-прокси на `127.0.0.1:8080` в `.env` используется `VECTOR_BASE_PUBLISHED_PORT=8081`, чтобы внешний порт `8080` оставался свободным под прокси.

Если этот прокси доступен только на loopback хоста, включается профиль `host-proxy-forwarder`; он не меняет Docker-сеть сервисов, а только делает локальный TCP-прокси хоста доступным контейнерам через `host.docker.internal`.

`bot` не публикует порты, потому что работает через Telegram long polling.

## Данные

- `bot_data` — Docker volume с SQLite базой бота.
- `chroma_data` — Docker volume с данными ChromaDB.
- `vector-base/meta/` — метаданные синхронизации Google Docs.
- `vector-base/artifacts/` — подготовленные RAG-артефакты и генераторы корпуса.
- `vector-base/secrets/google_service_account.json` — локальный Google service account key, игнорируется git.

## Эксплуатация И Безопасность

- `.env` и `vector-base/secrets/google_service_account.json` не должны попадать в git.
- ChromaDB не публикуется наружу; доступ к ней есть только у `vector-base` внутри Docker-сети.
- Для восстановления базы знаний можно использовать `make load-rag`, который загружает подготовленный JSONL-корпус.
- Для актуализации базы из Google Docs используется `make sync` или плановая синхронизация по `SYNC_INTERVAL_MINUTES`.
- В продакшене нужно резервировать Docker volume `chroma_data` и `bot_data`.
- Нужно следить за диском: ChromaDB и SQLite пишут данные в Docker volumes.
- При переносе на сервер достаточно перенести репозиторий, `.env`, Google service account и поднять стек через `make up`.

## Конфигурация

Единый runtime-файл: `.env`.

Шаблон: `.env.example`.

Ключевые переменные:

- `TELEGRAM_BOT_TOKEN`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_PROXY_URL`
- `TELEGRAM_PROXY_URL`
- `OPERATOR_CHAT_ID`
- `VECTOR_BASE_URL`
- `VECTOR_BASE_PUBLISHED_PORT`
- `GOOGLE_SERVICE_ACCOUNT_FILE`
- `GOOGLE_DOC_IDS`
- `CHROMA_HOST`
- `CHROMA_PORT`
- `CHROMA_COLLECTION_NAME`
- `SYNC_INTERVAL_MINUTES`

`requirements.txt` в `support-bot/` и `vector-base/` нужны Dockerfile для установки Python-зависимостей. Это не временная документация.

## Структура Репозитория

```text
support-bot-main/
├── docker-compose.yml
├── Makefile
├── README.md
├── AGENTS.md
├── docs/
│   ├── architecture.md
│   └── arch-rules.md
├── support-bot/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── system_prompt.txt
│   ├── app/
│   └── tests/
└── vector-base/
    ├── Dockerfile
    ├── .dockerignore
    ├── requirements.txt
    ├── app/
    ├── tests/
    ├── meta/
    ├── artifacts/
    └── secrets/
```

В подпроектах больше нет отдельных `README.md`, `AGENTS.md`, `.env.example`, `.gitignore` и `docs/`.

## Проверка

Из корня:

```bash
make test
make health
docker compose ps
```
