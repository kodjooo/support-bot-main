.PHONY: up down restart ps logs logs-bot logs-vector health test test-bot test-vector sync load-rag shell-bot shell-vector config

up:
	docker compose up -d --build

down:
	docker compose down

restart:
	docker compose restart

ps:
	docker compose ps

logs:
	docker compose logs -f

logs-bot:
	docker compose logs -f bot

logs-vector:
	docker compose logs -f vector-base

health:
	curl -sS "http://localhost:$$(docker compose port vector-base 8080 | sed 's/.*://')/health"

test:
	docker compose run --rm bot-tests
	docker compose run --rm vector-base-tests

test-bot:
	docker compose run --rm bot-tests

test-vector:
	docker compose run --rm vector-base-tests

sync:
	docker compose run --rm vector-base python -m app.sync_docs --force

load-rag:
	docker compose exec vector-base python -m app.load_rag_corpus artifacts/rag_corpus/rag_chunks.jsonl

shell-bot:
	docker compose run --rm bot sh

shell-vector:
	docker compose run --rm vector-base sh

config:
	docker compose config
