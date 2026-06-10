from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings, get_settings

__all__ = ["DocumentSnapshot", "GoogleDocsService"]


@dataclass
class DocumentSnapshot:
    """Снапшот документа Google Docs."""

    doc_id: str
    text: str
    modified_time: str


class GoogleDocsService:
    """Клиент для чтения документов и метаданных Google Docs/Drive с кешированием."""

    SCOPES = (
        "https://www.googleapis.com/auth/documents.readonly",
        "https://www.googleapis.com/auth/drive.metadata.readonly",
    )

    def __init__(
        self,
        *,
        settings: Optional[Settings] = None,
        cache_dir: Optional[Path] = None,
        credentials: Any = None,
        docs_client: Any = None,
        drive_client: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir or "meta")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._lock = Lock()
        self._last_call = 0.0

        self._credentials = credentials
        self._docs_client = docs_client
        self._drive_client = drive_client

        if self._credentials is None and (self._docs_client is None or self._drive_client is None):
            self._credentials = self._build_credentials()

        if self._docs_client is None:
            self._docs_client = self._build_docs_client()

        if self._drive_client is None:
            self._drive_client = self._build_drive_client()

        self._retryer = Retrying(
            stop=stop_after_attempt(self.settings.google_retry_attempts),
            wait=wait_exponential(multiplier=self.settings.google_retry_initial_delay),
            retry=retry_if_exception_type(HttpError),
            reraise=True,
        )

    def get_document_text(self, doc_id: str) -> str:
        """Возвращает текст документа."""

        def supplier() -> str:
            response = self._docs_client.documents().get(documentId=doc_id).execute()
            content = response.get("body", {}).get("content", [])
            return self._extract_text(content)

        return self._execute_with_retry(supplier)

    def get_document_modified_time(self, doc_id: str) -> str:
        """Возвращает значение modifiedTime из Drive."""

        def supplier() -> str:
            response = (
                self._drive_client.files()
                .get(fileId=doc_id, fields="modifiedTime")
                .execute()
            )
            return response["modifiedTime"]

        return self._execute_with_retry(supplier)

    def fetch_document(self, doc_id: str) -> DocumentSnapshot:
        """Возвращает текст и modifiedTime документа."""
        modified_time = self.get_document_modified_time(doc_id)
        text = self.get_document_text(doc_id)
        return DocumentSnapshot(doc_id=doc_id, text=text, modified_time=modified_time)

    def read_cached_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Читает сохранённые данные по документу."""
        path = self._cache_path(doc_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def persist_metadata(self, doc_id: str, modified_time: str) -> None:
        """Сохраняет метаданные документа на диск."""
        path = self._cache_path(doc_id)
        payload = {
            "doc_id": doc_id,
            "modifiedTime": modified_time,
            "updatedAt": datetime.now(timezone.utc).isoformat(),
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def needs_update(self, doc_id: str, modified_time: Optional[str] = None) -> bool:
        """Определяет, требуется ли обновление документа."""
        current_modified = modified_time or self.get_document_modified_time(doc_id)
        cached = self.read_cached_metadata(doc_id)
        if cached is None:
            return True
        return cached.get("modifiedTime") != current_modified

    def _execute_with_retry(self, supplier):
        def wrapped():
            self._throttle()
            return supplier()

        return self._retryer(wrapped)

    def _throttle(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            wait_for = self.settings.google_request_interval_seconds - elapsed
            if wait_for > 0:
                time.sleep(wait_for)
            self._last_call = time.monotonic()

    def _build_credentials(self):
        if self.settings.google_service_account_file:
            path = Path(self.settings.google_service_account_file)
            return service_account.Credentials.from_service_account_file(
                str(path),
                scopes=self.SCOPES,
            )

        info_secret = self.settings.google_service_account_info
        if info_secret is None:
            raise ValueError("Не удалось получить параметры сервисного аккаунта.")

        info = json.loads(info_secret.get_secret_value())
        return service_account.Credentials.from_service_account_info(
            info,
            scopes=self.SCOPES,
        )

    def _build_docs_client(self):
        return build(
            "docs",
            "v1",
            credentials=self._credentials,
            cache_discovery=False,
        )

    def _build_drive_client(self):
        return build(
            "drive",
            "v3",
            credentials=self._credentials,
            cache_discovery=False,
        )

    def _cache_path(self, doc_id: str) -> Path:
        safe_id = doc_id.replace("/", "_")
        return self.cache_dir / f"{safe_id}.json"

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Собирает текст из структуры документа."""
        fragments: list[str] = []
        for element in content or []:
            paragraph = element.get("paragraph") if isinstance(element, dict) else None
            if not paragraph:
                continue
            for item in paragraph.get("elements", []):
                text_run = item.get("textRun") if isinstance(item, dict) else None
                if not text_run:
                    continue
                value = text_run.get("content", "")
                if value:
                    fragments.append(value)
        return "".join(fragments).strip()
