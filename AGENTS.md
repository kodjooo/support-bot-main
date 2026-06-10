# AGENTS.md — support-bot-main

Основной проект: `support-bot-main`.

## Рабочий Контур

- Запускать и тестировать только из корня проекта.
- Основной compose-файл: `docker-compose.yml`.
- Основные команды: `make up`, `make ps`, `make logs-bot`, `make logs-vector`, `make test`, `make health`.
- `support-bot` и `vector-base` остаются отдельными сервисами, но работают в одном Docker Compose стеке.

## Документация

- Архитектура: `docs/architecture.md`.
- Архитектурные правила: `docs/arch-rules.md`.
- Общие команды и эксплуатация: `README.md`.
- В подпроектах нет отдельных `README.md`, `AGENTS.md`, `.env.example`, `.gitignore` и `docs/`; вся документация ведётся на корневом уровне.
- `plan.md` и `requirements.md` больше не используются.

## Правила Изменений

- При изменении поведения сервисов обновлять `docs/architecture.md`.
- При изменении общего запуска, сети, volumes или команд обновлять `README.md` и `docs/architecture.md`.
- При изменении зависимостей обновлять соответствующий `requirements.txt`; эти файлы нужны Docker-сборке.
- `.env`, Telegram token, OpenAI key и Google service account не отправлять в git.
- Комментарии в коде писать на русском языке.
- Для документации по библиотекам сначала использовать MCP Context7, затем интернет.
- Проверять изменения через `make test` и `make health`, если затронуты runtime или Docker.
