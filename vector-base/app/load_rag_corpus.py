from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from app.chroma_manager import VectorStoreGateway
from app.config import get_settings
from app.embeddings import EmbeddingService
from app.sync_docs import configure_logging

logger = logging.getLogger(__name__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not payload.get("id"):
                raise ValueError(f"В строке {line_number} нет id.")
            if not payload.get("embedding_text"):
                raise ValueError(f"В строке {line_number} нет embedding_text.")
            if not payload.get("answer"):
                raise ValueError(f"В строке {line_number} нет answer.")
            chunks.append(payload)
    return chunks


def _metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    base = dict(chunk.get("metadata") or {})
    base.update(
        {
            "doc_id": chunk.get("doc_id") or base.get("doc_id") or "",
            "doc_title": chunk.get("doc_title") or base.get("doc_title") or "",
            "chunk_id": chunk.get("id") or "",
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "section": base.get("section") or " > ".join(chunk.get("section_path") or []),
            "title": chunk.get("title") or base.get("title") or "",
            "block_type": chunk.get("block_type") or base.get("block_type") or "",
            "keywords": ", ".join(chunk.get("keywords") or [])[:1000],
            "question_intents": " | ".join(chunk.get("question_intents") or [])[:1500],
            "preview": (chunk.get("answer") or "")[:240],
        }
    )
    return {key: value for key, value in base.items() if value is not None}


def load_corpus(path: Path) -> int:
    settings = get_settings()
    chunks = _read_jsonl(path)
    logger.info("Загрузка RAG-корпуса из %s, блоков: %s.", path, len(chunks))

    embedding_service = EmbeddingService(settings=settings)
    embedding_results = embedding_service.embed_texts(chunk["embedding_text"] for chunk in chunks)
    if len(embedding_results) != len(chunks):
        raise RuntimeError("Количество эмбеддингов не совпало с количеством RAG-блоков.")

    vector_store = VectorStoreGateway(settings=settings)
    vector_store.replace_corpus(
        ids=[chunk["id"] for chunk in chunks],
        texts=[chunk["answer"] for chunk in chunks],
        embeddings=[item.embedding for item in embedding_results],
        metadatas=[_metadata(chunk) for chunk in chunks],
    )
    logger.info("RAG-корпус загружен в коллекцию %s.", settings.chroma_collection_name)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Загружает структурированный JSONL RAG-корпус в ChromaDB.")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("artifacts/rag_corpus/rag_chunks.jsonl"),
        help="Путь к rag_chunks.jsonl.",
    )
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    count = load_corpus(args.path)
    print(json.dumps({"status": "ok", "chunks": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
