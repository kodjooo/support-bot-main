from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from app.config import get_settings
from app.google_docs import GoogleDocsService
from app.openai_client import create_openai_client


SYSTEM_PROMPT = """Ты перерабатываешь русскоязычную базу знаний Sellerdata для RAG-поиска.
Твоя задача: сохранить все важные факты, правила, ограничения, названия экранов, кнопок,
полей, маркетплейсов и пользовательские сценарии, но разложить документ на независимые
контекстные блоки.

Каждый блок должен отвечать на один устойчивый поисковый интент. Не придумывай факты.
Если в исходнике нет ответа, не добавляй его. Пиши на русском.
"""

USER_PROMPT_TEMPLATE = """Исходный Google Docs документ:
Название: {title}
doc_id: {doc_id}
modifiedTime: {modified_time}

Текст документа:
<<<
{text}
>>>

Верни строго JSON-объект:
{{
  "document_summary": "краткое описание документа в 1-2 предложения",
  "blocks": [
    {{
      "section_path": ["Раздел", "Подраздел"],
      "block_type": "overview|how_to|field_reference|calculation_rule|limitation|troubleshooting|faq|comparison|data_freshness|notification|integration",
      "title": "краткий заголовок блока",
      "question_intents": ["как пользователь может спросить это другими словами"],
      "keywords": ["термины, поля, кнопки, маркетплейсы, синонимы"],
      "answer": "самодостаточный ответ/описание без ссылок 'выше' и 'ниже'",
      "source_facts": ["короткие проверочные факты из исходника, которые были использованы"]
    }}
  ]
}}

Требования:
- Делай блоки самодостаточными: в answer должен быть понятен экран/раздел и условие применения.
- Не склеивай разные сценарии в один блок, если пользователь будет искать их разными вопросами.
- Не теряй числа, периоды, условия, исключения, названия колонок, кнопок и статусов.
- Добавляй синонимы и разговорные формулировки в question_intents и keywords.
- Если раздел большой, создай больше блоков, а не один длинный.
- answer держи обычно 80-220 слов, но можно длиннее, если без этого теряются правила.
"""


@dataclass
class SourceDocument:
    doc_id: str
    title: str
    modified_time: str
    text: str


def compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def stable_slug(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "-", value.lower()).strip("-")
    return slug[:80] or "document"


def read_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def fetch_sources(output_dir: Path) -> list[SourceDocument]:
    settings = get_settings()
    docs_service = GoogleDocsService(settings=settings)
    sources_dir = output_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)

    documents: list[SourceDocument] = []
    for doc_id in settings.google_doc_ids:
        metadata = (
            docs_service._drive_client.files()  # noqa: SLF001 - artifact generator needs Drive title.
            .get(fileId=doc_id, fields="name,modifiedTime")
            .execute()
        )
        snapshot = docs_service.fetch_document(doc_id)
        title = compact_text(metadata.get("name") or doc_id)
        source = SourceDocument(
            doc_id=doc_id,
            title=title,
            modified_time=metadata.get("modifiedTime") or snapshot.modified_time,
            text=snapshot.text,
        )
        documents.append(source)

        source_path = sources_dir / f"{stable_slug(title)}__{doc_id}.txt"
        source_path.write_text(source.text, encoding="utf-8")

    return documents


def rewrite_document(client: OpenAI, source: SourceDocument, model: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    title=source.title,
                    doc_id=source.doc_id,
                    modified_time=source.modified_time,
                    text=source.text,
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    return read_json_object(content)


def normalize_blocks(source: SourceDocument, rewritten: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = rewritten.get("blocks") or []
    normalized: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        section_path = [compact_text(str(item)) for item in block.get("section_path", []) if compact_text(str(item))]
        title = compact_text(str(block.get("title") or f"Блок {index + 1}"))
        answer = compact_text(str(block.get("answer") or ""))
        intents = [
            compact_text(str(item))
            for item in block.get("question_intents", [])
            if compact_text(str(item))
        ]
        keywords = [
            compact_text(str(item))
            for item in block.get("keywords", [])
            if compact_text(str(item))
        ]
        source_facts = [
            compact_text(str(item))
            for item in block.get("source_facts", [])
            if compact_text(str(item))
        ]
        if not answer:
            continue

        searchable_text = "\n".join(
            part
            for part in [
                f"Документ: {source.title}",
                f"Раздел: {' > '.join(section_path)}" if section_path else "",
                f"Заголовок: {title}",
                f"Вопросы: {'; '.join(intents)}" if intents else "",
                f"Ключевые слова: {', '.join(keywords)}" if keywords else "",
                f"Ответ: {answer}",
            ]
            if part
        )
        content_hash = hashlib.sha256(searchable_text.encode("utf-8")).hexdigest()[:16]
        normalized.append(
            {
                "id": f"{source.doc_id}:{index:04d}:{content_hash}",
                "doc_id": source.doc_id,
                "doc_title": source.title,
                "doc_modified_time": source.modified_time,
                "chunk_index": index,
                "section_path": section_path,
                "block_type": compact_text(str(block.get("block_type") or "faq")),
                "title": title,
                "question_intents": intents,
                "keywords": keywords,
                "answer": answer,
                "source_facts": source_facts,
                "embedding_text": searchable_text,
                "metadata": {
                    "doc_id": source.doc_id,
                    "doc_title": source.title,
                    "section": " > ".join(section_path),
                    "title": title,
                    "block_type": compact_text(str(block.get("block_type") or "faq")),
                    "keywords": ", ".join(keywords[:20]),
                    "preview": answer[:240],
                },
            }
        )
    return normalized


def write_outputs(output_dir: Path, sources: list[SourceDocument], chunks: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "rag_chunks.jsonl"
    json_path = output_dir / "rag_chunks.json"
    manifest_path = output_dir / "manifest.json"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    by_doc: dict[str, int] = {}
    for chunk in chunks:
        by_doc[chunk["doc_id"]] = by_doc.get(chunk["doc_id"], 0) + 1

    manifest = {
        "format": "sellerdata_rag_corpus_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_documents": [asdict(source) | {"text_chars": len(source.text)} for source in sources],
        "chunk_count": len(chunks),
        "chunk_count_by_doc": by_doc,
        "files": {
            "jsonl": str(jsonl_path),
            "json": str(json_path),
            "sources_dir": str(output_dir / "sources"),
        },
        "usage": {
            "primary_index_file": "rag_chunks.jsonl",
            "embedding_field": "embedding_text",
            "metadata_field": "metadata",
            "answer_field": "answer",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает структурированный RAG-корпус из Google Docs.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rag_corpus"))
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument(
        "--reuse-rewritten",
        action="store_true",
        help="Использовать уже сохранённые rewritten/*.json и не вызывать модель повторно.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = fetch_sources(args.output_dir)
    client = create_openai_client(get_settings(), timeout=120.0, max_retries=2)

    all_chunks: list[dict[str, Any]] = []
    rewritten_dir = args.output_dir / "rewritten"
    rewritten_dir.mkdir(parents=True, exist_ok=True)
    for source in sources:
        rewritten_path = rewritten_dir / f"{stable_slug(source.title)}__{source.doc_id}.json"
        if args.reuse_rewritten and rewritten_path.exists():
            rewritten = json.loads(rewritten_path.read_text(encoding="utf-8"))
        else:
            rewritten = rewrite_document(client, source, args.model)
            rewritten_path.write_text(
                json.dumps(rewritten, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        all_chunks.extend(normalize_blocks(source, rewritten))

    write_outputs(args.output_dir, sources, all_chunks)
    print(
        json.dumps(
            {
                "documents": len(sources),
                "chunks": len(all_chunks),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
