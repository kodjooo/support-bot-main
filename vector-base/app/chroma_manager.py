from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Структурированный результат поиска по базе знаний."""

    text: str
    metadata: dict
    distance: float | None = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)


class VectorStoreGateway:
    """Инкапсулирует работу с ChromaDB."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or self._build_client()
        self._collection: Collection | None = None

    def replace_document(
        self,
        *,
        doc_id: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Iterable[dict] | None = None,
    ) -> None:
        """Удаляет старые записи документа и добавляет новые чанки."""
        if not texts:
            logger.warning("Получен пустой набор текстов для документа %s, запись пропущена.", doc_id)
            return

        collection = self._get_collection()
        logger.debug("Удаление существующих записей документа %s из коллекции.", doc_id)
        collection.delete(where={"doc_id": doc_id})

        ids = [f"{doc_id}-{index}" for index in range(len(texts))]
        metadata_payload = metadatas or [{"doc_id": doc_id, "chunk": index} for index in range(len(texts))]

        logger.debug("Добавление %s чанков для документа %s.", len(texts), doc_id)
        collection.add(
            ids=ids,
            documents=list(texts),
            embeddings=list(embeddings),
            metadatas=list(metadata_payload),
        )

    def replace_corpus(
        self,
        *,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict],
    ) -> None:
        """Полностью заменяет содержимое коллекции готовым RAG-корпусом."""
        if not texts:
            logger.warning("Получен пустой RAG-корпус, коллекция не изменена.")
            return

        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("Размеры ids, texts, embeddings и metadatas должны совпадать.")

        logger.info("Пересоздание коллекции %s для RAG-корпуса.", self.settings.chroma_collection_name)
        try:
            self._client.delete_collection(self.settings.chroma_collection_name)
        except Exception:  # noqa: BLE001 - разные версии Chroma возвращают разные типы ошибок.
            logger.debug("Коллекция %s ещё не существовала.", self.settings.chroma_collection_name)
        self._collection = self._client.get_or_create_collection(self.settings.chroma_collection_name)
        self._collection.add(
            ids=list(ids),
            documents=list(texts),
            embeddings=list(embeddings),
            metadatas=list(metadatas),
        )

    def delete_document(self, doc_id: str) -> None:
        logger.debug("Удаление документа %s из коллекции.", doc_id)
        self._get_collection().delete(where={"doc_id": doc_id})

    def query(self, embedding: Sequence[float], *, limit: int = 3) -> dict:
        logger.debug("Поиск релевантных документов (limit=%s).", limit)
        return self._get_collection().query(query_embeddings=[embedding], n_results=limit)

    def search(
        self,
        embedding: Sequence[float],
        *,
        query: str,
        limit: int = 3,
        candidate_limit: int | None = None,
        min_score: float = 0.0,
        keyword_limit: int = 2000,
    ) -> list[SearchResult]:
        """Ищет чанки и переранжирует их с учётом точных совпадений слов."""
        raw_limit = candidate_limit or limit
        query_result = self.query(embedding, limit=raw_limit)
        results = self._merge_results(
            self._flatten_query_result(query_result),
            self._keyword_candidates(keyword_limit=keyword_limit),
        )
        ranked = self._rerank_results(query, results)
        if any(item.keyword_score > 0 for item in ranked):
            ranked = [item for item in ranked if item.keyword_score > 0]
        filtered = [item for item in ranked if item.score >= min_score]
        return filtered[:limit]

    def _flatten_query_result(self, query_result: dict) -> list[SearchResult]:
        documents = query_result.get("documents") or []
        metadatas = query_result.get("metadatas") or []
        distances = query_result.get("distances") or []

        results: list[SearchResult] = []
        for group_index, group in enumerate(documents):
            metadata_group = metadatas[group_index] if group_index < len(metadatas) else []
            distance_group = distances[group_index] if group_index < len(distances) else []
            for item_index, text in enumerate(group):
                if not text:
                    continue
                distance = distance_group[item_index] if item_index < len(distance_group) else None
                metadata = metadata_group[item_index] if item_index < len(metadata_group) else {}
                semantic_score = self._semantic_score(distance)
                results.append(
                    SearchResult(
                        text=text,
                        metadata=metadata or {},
                        distance=distance,
                        semantic_score=semantic_score,
                        score=semantic_score,
                    ),
                )
        return results

    def _rerank_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = _extract_terms(query)
        query_intent = _detect_query_intent(query)
        for result in results:
            text_terms = set(_extract_terms(result.text))
            metadata_text = " ".join(
                str(result.metadata.get(key) or "")
                for key in ("section", "title", "preview")
            )
            metadata_terms = set(_extract_terms(metadata_text))
            matched_text = {term for term in query_terms if term in text_terms}
            matched_metadata = {term for term in query_terms if term in metadata_terms}
            matched = sorted(matched_text | matched_metadata)
            result.matched_terms = matched
            result.keyword_score = _keyword_score(
                query_intent=query_intent,
                query_terms=query_terms,
                matched_text=matched_text,
                matched_metadata=matched_metadata,
                result=result,
            )
            result.score = result.semantic_score + result.keyword_score
            if matched_text:
                result.text = _focused_excerpt(result.text, matched_text)
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _keyword_candidates(self, *, keyword_limit: int) -> list[SearchResult]:
        payload = self._get_collection().get(
            include=["documents", "metadatas"],
            limit=keyword_limit,
        )
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []

        results: list[SearchResult] = []
        for index, text in enumerate(documents):
            if not text:
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            results.append(SearchResult(text=text, metadata=metadata or {}))
        return results

    @staticmethod
    def _merge_results(*groups: list[SearchResult]) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        for group in groups:
            for item in group:
                existing = merged.get(item.text)
                if existing is None or item.semantic_score > existing.semantic_score:
                    merged[item.text] = item
        return list(merged.values())

    @staticmethod
    def _semantic_score(distance: float | None) -> float:
        if distance is None:
            return 0.0
        if not math.isfinite(distance):
            return 0.0
        return 1 / (1 + max(distance, 0))

    def _get_collection(self) -> Collection:
        if self._collection is None:
            logger.debug(
                "Получение коллекции %s на %s:%s.",
                self.settings.chroma_collection_name,
                self.settings.chroma_host,
                self.settings.chroma_port,
            )
            self._collection = self._client.get_or_create_collection(self.settings.chroma_collection_name)
        return self._collection

    def _build_client(self) -> ClientAPI:
        logger.debug(
            "Инициализация клиента ChromaDB на %s:%s.",
            self.settings.chroma_host,
            self.settings.chroma_port,
        )
        return chromadb.HttpClient(
            host=self.settings.chroma_host,
            port=self.settings.chroma_port,
        )


def _extract_terms(text: str) -> list[str]:
    terms = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
    stop_words = {
        "где",
        "могу",
        "можно",
        "увидеть",
        "посмотреть",
        "покажи",
        "найти",
        "какой",
        "какая",
        "какие",
    }
    extracted = [
        _normalize_term(term)
        for term in terms
        if (len(term) >= 4 or term in {"wb", "вб"}) and term not in stop_words
    ]
    return _expand_query_terms(text, extracted)


def _expand_query_terms(text: str, terms: list[str]) -> list[str]:
    """Добавляет общеупотребимые синонимы и сокращения без привязки к одному кейсу."""
    normalized = text.lower()
    expanded = list(terms)

    aliases = {
        "wb": ("wildberries", "вайлдберриз"),
        "вб": ("wildberries", "вайлдберриз"),
        "wildberries": ("wb", "вб", "вайлдберриз"),
        "вайлдберриз": ("wildberries", "wb", "вб"),
        "ozon": ("озон",),
        "озон": ("ozon",),
    }
    for term in terms:
        expanded.extend(aliases.get(term, ()))

    if any(marker in normalized for marker in ("не сход", "не совпад", "расхожден", "свер")):
        expanded.extend(("сверк", "совпад", "расхожден", "задержк"))
    if any(term.startswith("выплат") for term in terms):
        expanded.extend(("выплат", "оплат", "перечислен", "задержк"))
    if "отчет" in normalized or "отчёт" in normalized:
        expanded.extend(("отчет", "отчёт", "сверк"))

    deduplicated: list[str] = []
    seen: set[str] = set()
    for term in expanded:
        normalized_term = _normalize_term(term)
        if normalized_term and normalized_term not in seen:
            seen.add(normalized_term)
            deduplicated.append(normalized_term)
    return deduplicated


def _normalize_term(term: str) -> str:
    endings = (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "иях",
        "ах",
        "ях",
        "ую",
        "юю",
        "ая",
        "яя",
        "ое",
        "ее",
        "ые",
        "ие",
        "ый",
        "ий",
        "ой",
        "ам",
        "ям",
        "ом",
        "ем",
        "ов",
        "ев",
        "ей",
        "ий",
        "ия",
        "ие",
        "ии",
        "я",
        "а",
        "у",
        "ю",
        "ы",
        "и",
        "е",
    )
    for ending in endings:
        if len(term) > len(ending) + 4 and term.endswith(ending):
            normalized = term[: -len(ending)]
            return normalized[:-1] if normalized.endswith("и") and len(normalized) > 5 else normalized
    return term


def _detect_query_intent(query: str) -> str:
    normalized = query.lower()
    reconciliation_markers = (
        "не сход",
        "не совпад",
        "расхожден",
        "свер",
        "сравнить",
        "почему выплаты",
    )
    if any(marker in normalized for marker in reconciliation_markers):
        return "reconciliation"
    navigation_markers = (
        "где",
        "куда",
        "как найти",
        "как посмотреть",
        "где посмотреть",
        "где увидеть",
        "как открыть",
        "куда нажать",
    )
    if any(marker in normalized for marker in navigation_markers):
        return "navigation"
    if any(marker in normalized for marker in ("как", "почему", "зачем", "что делать")):
        return "instruction"
    return "general"


def _keyword_score(
    *,
    query_intent: str,
    query_terms: list[str],
    matched_text: set[str],
    matched_metadata: set[str],
    result: SearchResult,
) -> float:
    if not query_terms:
        return 0.0

    text = result.text.lower()
    matched = matched_text | matched_metadata
    score = min(len(matched_text) * 0.45 + len(matched_metadata) * 0.25, 2.0)
    coverage = len(matched) / len(set(query_terms))
    score += min(coverage * 0.35, 0.35)

    if query_intent == "navigation":
        score += _navigation_score(text)
    elif query_intent == "instruction":
        score += _instruction_score(text)
    elif query_intent == "reconciliation":
        score += _reconciliation_score(text)

    score += _term_position_score(query_terms, text)
    if result.metadata.get("section") or result.metadata.get("title"):
        score += 0.15

    if matched and not _has_actionable_signal(text):
        score -= 0.25

    return max(score, 0.0)


def _navigation_score(text: str) -> float:
    score = 0.0
    action_markers = (
        "нажать",
        "нажмите",
        "клик",
        "при клике",
        "открыть",
        "откроется",
        "перейти",
        "выбрать",
        "раскрыть",
        "раскрывается",
        "расположен",
        "расположена",
        "находится",
        "отображается",
    )
    ui_markers = (
        "раздел",
        "вкладка",
        "кнопка",
        "колонка",
        "строка",
        "таблица",
        "фильтр",
        "иконка",
        "плашка",
        "правом верхнем",
        "левом нижнем",
    )
    if any(marker in text for marker in action_markers):
        score += 0.45
    if any(marker in text for marker in ui_markers):
        score += 0.35
    if "инструкция" in text:
        score += 0.15
    return score


def _instruction_score(text: str) -> float:
    score = 0.0
    if any(marker in text for marker in ("необходимо", "нужно", "важно", "для того чтобы", "после этого")):
        score += 0.35
    if any(marker in text for marker in ("проверить", "выбрать", "указать", "загрузить", "добавить", "скачать")):
        score += 0.35
    return score


def _reconciliation_score(text: str) -> float:
    score = 0.0
    if any(marker in text for marker in ("свер", "сравнить", "не совпад", "расхожден")):
        score += 0.45
    if any(marker in text for marker in ("не совпад", "может не совпад", "не сход")):
        score += 0.55
    if "выплат" in text:
        score += 0.45
    if "сумма выплат" in text:
        score += 0.3
    if any(marker in text for marker in ("задержк", "период", "отчет", "отчёт")):
        score += 0.3
    if "задержк" in text:
        score += 0.35
    if any(marker in text for marker in ("wildberries", "ozon", "вайлдберриз", "озон")):
        score += 0.2
    return score


def _has_actionable_signal(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "наж",
            "клик",
            "откры",
            "перей",
            "выбер",
            "раскр",
            "раздел",
            "вкладк",
            "кнопк",
            "колонк",
            "строк",
            "таблиц",
            "фильтр",
        )
    )


def _term_position_score(query_terms: list[str], text: str) -> float:
    terms = _extract_terms(text)
    if not terms:
        return 0.0

    positions = [
        index
        for index, term in enumerate(terms)
        if term in query_terms
    ]
    if not positions:
        return 0.0

    first = min(positions)
    span = max(positions) - first if len(positions) > 1 else len(terms)
    early_score = max(0.0, 1 - first / max(len(terms), 1)) * 0.3
    density_score = max(0.0, 1 - span / max(len(terms), 1)) * 0.25
    return early_score + density_score


def _focused_excerpt(text: str, matched_terms: set[str], *, max_sentences: int = 4) -> str:
    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return text

    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        terms = set(_extract_terms(sentence))
        score = len(terms & matched_terms)
        if score:
            scored.append((score, -index, sentence))

    if not scored:
        return " ".join(sentences[:max_sentences])

    _, negative_index, _ = max(scored)
    index = -negative_index
    start = max(index - 1, 0)
    end = min(start + max_sentences, len(sentences))
    if end - start < max_sentences:
        start = max(end - max_sentences, 0)
    return " ".join(sentences[start:end])


def _split_sentences(text: str) -> list[str]:
    compact = " ".join(text.split())
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«])", compact)
        if part.strip()
    ]
