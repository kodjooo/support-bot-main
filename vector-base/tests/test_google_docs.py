from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Iterable, List
from unittest import mock

from googleapiclient.errors import HttpError

from app.google_docs import DocumentSnapshot, GoogleDocsService


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


def make_settings(**overrides: Any) -> SimpleNamespace:
    payload = {
        "google_service_account_file": None,
        "google_service_account_info": _Secret('{"type": "service_account"}'),
        "google_request_interval_seconds": 0.0001,
        "google_retry_attempts": 3,
        "google_retry_initial_delay": 0.0001,
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class FakeRequest:
    def __init__(self, outcome: Any) -> None:
        self._outcome = outcome

    def execute(self) -> Any:
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class FakeDocsClient:
    def __init__(self, outcomes: Iterable[Any]) -> None:
        self._outcomes: List[Any] = list(outcomes)
        self.calls = 0

    def documents(self) -> "FakeDocsClient":
        return self

    def get(self, documentId: str) -> FakeRequest:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        return FakeRequest(outcome)


class FakeDriveClient:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls = 0

    def files(self) -> "FakeDriveClient":
        return self

    def get(self, fileId: str, fields: str) -> FakeRequest:
        self.calls += 1
        return FakeRequest(dict(self.outcome))


def test_get_document_text_retries_on_http_error(tmp_path) -> None:
    http_error = HttpError(mock.Mock(status=500), b"error")
    doc_body = {
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [
                            {"textRun": {"content": "Привет\n"}},
                        ],
                    },
                },
            ],
        },
    }

    docs_client = FakeDocsClient([http_error, doc_body])
    drive_client = FakeDriveClient({"modifiedTime": "2024-06-01T00:00:00Z"})
    service = GoogleDocsService(
        settings=make_settings(),
        cache_dir=tmp_path,
        credentials=object(),
        docs_client=docs_client,
        drive_client=drive_client,
    )

    text = service.get_document_text("doc-1")

    assert text == "Привет"
    assert docs_client.calls == 2


def test_needs_update_uses_cached_metadata(tmp_path) -> None:
    docs_client = FakeDocsClient(
        [
            {
                "body": {
                    "content": [
                        {
                            "paragraph": {
                                "elements": [
                                    {"textRun": {"content": "текст\n"}},
                                ],
                            },
                        },
                    ],
                },
            },
        ],
    )
    drive_client = FakeDriveClient({"modifiedTime": "2024-06-01T00:00:00Z"})
    service = GoogleDocsService(
        settings=make_settings(),
        cache_dir=tmp_path,
        credentials=object(),
        docs_client=docs_client,
        drive_client=drive_client,
    )

    assert service.needs_update("doc-1") is True

    service.persist_metadata("doc-1", "2024-06-01T00:00:00Z")
    assert service.needs_update("doc-1") is False

    drive_client.outcome = {"modifiedTime": "2024-06-02T00:00:00Z"}
    assert service.needs_update("doc-1") is True


def test_fetch_document_returns_snapshot(tmp_path) -> None:
    doc_body = {
        "body": {
            "content": [
                {
                    "paragraph": {
                        "elements": [
                            {"textRun": {"content": "Первый абзац\n"}},
                            {"textRun": {"content": "Вторая строка"}},
                        ],
                    },
                },
            ],
        },
    }
    docs_client = FakeDocsClient([doc_body])
    drive_client = FakeDriveClient({"modifiedTime": "2024-06-01T00:00:00Z"})
    service = GoogleDocsService(
        settings=make_settings(),
        cache_dir=tmp_path,
        credentials=object(),
        docs_client=docs_client,
        drive_client=drive_client,
    )

    snapshot = service.fetch_document("doc-1")

    assert isinstance(snapshot, DocumentSnapshot)
    assert snapshot.text.startswith("Первый абзац")
    assert snapshot.modified_time == "2024-06-01T00:00:00Z"
