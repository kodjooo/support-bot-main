import os

import pytest

from app.config import get_settings, reload_settings


def _set_common_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("SEARCH_TOP_K", "3")
    monkeypatch.setenv("SEARCH_CANDIDATE_MULTIPLIER", "4")
    monkeypatch.setenv("SEARCH_MIN_SCORE", "0.05")
    monkeypatch.setenv("SEARCH_KEYWORD_LIMIT", "2000")
    monkeypatch.setenv("GOOGLE_DOC_IDS", "doc-a,doc-b")
    monkeypatch.setenv("CHROMA_HOST", "chroma")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "knowledge")
    monkeypatch.setenv("EMBEDDING_CHUNK_SIZE", "500")
    monkeypatch.setenv("EMBEDDING_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("SYNC_INTERVAL_MINUTES", "20")
    monkeypatch.setenv("API_PORT", "8080")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


def test_get_settings_reads_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    reload_settings()
    _set_common_env(monkeypatch)
    service_account_path = tmp_path / "service.json"
    service_account_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", os.fspath(service_account_path))

    settings = get_settings()

    assert settings.openai_api_key.get_secret_value() == "test-openai-key"
    assert settings.google_doc_ids == ["doc-a", "doc-b"]
    assert settings.google_service_account_file == service_account_path
    assert settings.embedding_chunk_size == 500
    assert settings.embedding_chunk_overlap == 100
    assert settings.api_port == 8080
    assert settings.search_top_k == 3
    assert settings.search_candidate_multiplier == 4
    assert settings.search_min_score == 0.05
    assert settings.search_keyword_limit == 2000


def test_missing_service_account_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    reload_settings()
    _set_common_env(monkeypatch)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_FILE", raising=False)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_INFO", raising=False)
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_INFO", "")

    with pytest.raises(ValueError):
        get_settings()
