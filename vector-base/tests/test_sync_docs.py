from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from app.embeddings import EmbeddingResult
from app.google_docs import DocumentSnapshot
from app.sync_docs import SyncOrchestrator, _semantic_chunks


@dataclass
class StubSettings:
    google_doc_ids: List[str]
    embedding_chunk_size: int = 3
    embedding_chunk_overlap: int = 1
    log_level: str = "INFO"


class DocsStub:
    def __init__(self, snapshots: Dict[str, DocumentSnapshot], needs_update: bool) -> None:
        self.snapshots = snapshots
        self._needs_update = needs_update
        self.persisted = []
        self.fetch_calls = 0
        self.deleted = []

    def needs_update(self, doc_id: str) -> bool:
        return self._needs_update

    def fetch_document(self, doc_id: str) -> DocumentSnapshot:
        self.fetch_calls += 1
        return self.snapshots[doc_id]

    def persist_metadata(self, doc_id: str, modified_time: str) -> None:
        self.persisted.append((doc_id, modified_time))


class EmbeddingStub:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        self.calls.append(list(texts))
        return [
            EmbeddingResult(text=chunk, embedding=[float(index)])
            for index, chunk in enumerate(texts)
        ]


class VectorStub:
    def __init__(self) -> None:
        self.replacements = []
        self.deleted = []

    def replace_document(self, *, doc_id: str, texts, embeddings, metadatas) -> None:
        self.replacements.append(
            {
                "doc_id": doc_id,
                "texts": list(texts),
                "embeddings": list(embeddings),
                "metadatas": list(metadatas),
            },
        )

    def delete_document(self, doc_id: str) -> None:
        self.deleted.append(doc_id)


def test_sync_skips_when_not_modified():
    doc_id = "doc-1"
    settings = StubSettings(google_doc_ids=[doc_id])
    docs = DocsStub(
        snapshots={doc_id: DocumentSnapshot(doc_id=doc_id, text="text", modified_time="2024")},
        needs_update=False,
    )
    embeddings = EmbeddingStub()
    vector = VectorStub()

    orchestrator = SyncOrchestrator(
        settings=settings,
        docs_service=docs,
        embedding_service=embeddings,
        vector_gateway=vector,
    )

    results = orchestrator.sync_documents()

    assert results[0].status == "skipped"
    assert docs.fetch_calls == 0
    assert not vector.replacements


def test_sync_updates_document_and_stores_metadata():
    doc_id = "doc-2"
    settings = StubSettings(google_doc_ids=[doc_id], embedding_chunk_size=4, embedding_chunk_overlap=1)
    snapshot = DocumentSnapshot(doc_id=doc_id, text="alpha beta gamma delta", modified_time="2024-06-01T00:00:00Z")
    docs = DocsStub(snapshots={doc_id: snapshot}, needs_update=True)
    embeddings = EmbeddingStub()
    vector = VectorStub()

    orchestrator = SyncOrchestrator(
        settings=settings,
        docs_service=docs,
        embedding_service=embeddings,
        vector_gateway=vector,
    )

    results = orchestrator.sync_documents()

    assert results[0].status == "updated"
    assert docs.persisted == [(doc_id, "2024-06-01T00:00:00Z")]
    assert len(vector.replacements) == 1
    payload = vector.replacements[0]
    assert payload["doc_id"] == doc_id
    assert payload["metadatas"][0]["doc_id"] == doc_id
    assert "section" in payload["metadatas"][0]
    assert "title" in payload["metadatas"][0]


def test_sync_deletes_document_when_no_text():
    doc_id = "doc-empty"
    settings = StubSettings(google_doc_ids=[doc_id])
    snapshot = DocumentSnapshot(doc_id=doc_id, text="", modified_time="2024-07-01T00:00:00Z")
    docs = DocsStub(snapshots={doc_id: snapshot}, needs_update=True)
    embeddings = EmbeddingStub()
    vector = VectorStub()

    orchestrator = SyncOrchestrator(
        settings=settings,
        docs_service=docs,
        embedding_service=embeddings,
        vector_gateway=vector,
    )

    results = orchestrator.sync_documents()

    assert results[0].status == "deleted"
    assert vector.deleted == [doc_id]
    assert not embeddings.calls
    assert docs.persisted == [(doc_id, "2024-07-01T00:00:00Z")]


def test_semantic_chunks_keep_sentence_boundaries():
    text = (
        "Раздел «Дэшборд» показывает продажи и прибыль. "
        "Строка «Удержания» раскрывается и показывает комиссию, доставку и хранение. "
        "Вкладка «Заказы» показывает статусы заказов и детализацию."
    )

    chunks = _semantic_chunks(text, max_tokens=12, overlap=0)

    assert chunks[0].startswith("Раздел «Дэшборд»")
    assert chunks[1].startswith("Строка «Удержания»")
    assert chunks[2].startswith("Вкладка «Заказы»")
