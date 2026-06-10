from __future__ import annotations

import json

from app.embeddings import EmbeddingResult
from app.load_rag_corpus import _metadata, _read_jsonl, load_corpus


def test_read_jsonl_validates_required_fields(tmp_path):
    path = tmp_path / "rag.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "chunk-1",
                "embedding_text": "Документ: Товары\nОтвет: как загрузить себестоимость",
                "answer": "Чтобы загрузить себестоимость, откройте раздел «Товары».",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    chunks = _read_jsonl(path)

    assert chunks[0]["id"] == "chunk-1"


def test_metadata_flattens_lists_for_chroma():
    chunk = {
        "id": "doc-1:0001",
        "doc_id": "doc-1",
        "doc_title": "Раздел «Товары»",
        "chunk_index": 1,
        "section_path": ["Товары", "Себестоимость"],
        "block_type": "how_to",
        "title": "Загрузка себестоимости",
        "keywords": ["товары", "себестоимость"],
        "question_intents": ["как загрузить себестоимость"],
        "answer": "Откройте раздел «Товары» и загрузите таблицу.",
    }

    metadata = _metadata(chunk)

    assert metadata["section"] == "Товары > Себестоимость"
    assert metadata["keywords"] == "товары, себестоимость"
    assert metadata["question_intents"] == "как загрузить себестоимость"


def test_load_corpus_embeds_embedding_text_and_stores_answer(tmp_path, monkeypatch):
    path = tmp_path / "rag.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "chunk-1",
                "doc_id": "doc-1",
                "doc_title": "Раздел «Дашборд»",
                "chunk_index": 0,
                "section_path": ["Дашборд"],
                "block_type": "faq",
                "title": "Удержания",
                "keywords": ["удержания"],
                "question_intents": ["где посмотреть удержания"],
                "embedding_text": "Документ: Дашборд\nВопросы: где посмотреть удержания\nОтвет: удержания в плитках",
                "answer": "Удержания раскрываются в детализации плашки дашборда.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    class SettingsStub:
        chroma_collection_name = "knowledge"

    class EmbeddingsStub:
        calls = []

        def __init__(self, settings):
            self.settings = settings

        def embed_texts(self, texts):
            self.calls.append(list(texts))
            return [EmbeddingResult(text=self.calls[0][0], embedding=[0.1])]

    class VectorStub:
        payload = None

        def __init__(self, settings):
            self.settings = settings

        def replace_corpus(self, *, ids, texts, embeddings, metadatas):
            self.payload = {
                "ids": ids,
                "texts": texts,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
            VectorStub.payload = self.payload

    monkeypatch.setattr("app.load_rag_corpus.get_settings", lambda: SettingsStub())
    monkeypatch.setattr("app.load_rag_corpus.EmbeddingService", EmbeddingsStub)
    monkeypatch.setattr("app.load_rag_corpus.VectorStoreGateway", VectorStub)

    count = load_corpus(path)

    assert count == 1
    assert EmbeddingsStub.calls == [
        ["Документ: Дашборд\nВопросы: где посмотреть удержания\nОтвет: удержания в плитках"],
    ]
    assert VectorStub.payload["texts"] == ["Удержания раскрываются в детализации плашки дашборда."]
