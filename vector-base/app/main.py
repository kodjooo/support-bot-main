"""Точка входа: HTTP API + планировщик синхронизации документов."""
from __future__ import annotations

import logging

import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler

from app.api import app
from app.config import get_settings
from app.sync_docs import SyncOrchestrator, configure_logging


def _run_sync() -> None:
    """Запускает синхронизацию документов (вызывается планировщиком)."""
    logger = logging.getLogger(__name__)
    logger.info("Плановая синхронизация документов запущена.")
    try:
        orchestrator = SyncOrchestrator()
        results = orchestrator.sync_documents()
        for result in results:
            if result.status == "failed":
                logger.error("Документ %s: ошибка — %s", result.doc_id, result.error)
            else:
                logger.info("Документ %s: %s (чанков: %s)", result.doc_id, result.status, result.chunks)
    except Exception:
        logger.exception("Ошибка во время плановой синхронизации.")


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)

    # Планировщик синхронизации в фоновом потоке
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _run_sync,
        trigger="interval",
        minutes=settings.sync_interval_minutes,
        id="sync_docs",
    )
    scheduler.start()
    logger.info(
        "Планировщик синхронизации запущен (каждые %s минут).",
        settings.sync_interval_minutes,
    )

    # Запускаем HTTP API
    logger.info("Запуск HTTP API на порту %s.", settings.api_port)
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
