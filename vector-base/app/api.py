"""HTTP API для поиска по векторной базе знаний."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.chroma_manager import VectorStoreGateway
from app.config import get_settings
from app.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

app = FastAPI(title="Vector Search API", docs_url=None, redoc_url=None)

# Сервисы инициализируются один раз при старте
_embedding_service: EmbeddingService | None = None
_vector_store: VectorStoreGateway | None = None


def _get_services() -> tuple[EmbeddingService, VectorStoreGateway]:
    global _embedding_service, _vector_store
    if _embedding_service is None:
        settings = get_settings()
        _embedding_service = EmbeddingService(settings=settings)
        _vector_store = VectorStoreGateway(settings=settings)
    return _embedding_service, _vector_store


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResultResponse(BaseModel):
    text: str
    metadata: dict
    distance: float | None = None
    score: float
    semantic_score: float
    keyword_score: float
    matched_terms: list[str]


class SearchResponse(BaseModel):
    chunks: list[str]
    results: list[SearchResultResponse] = Field(default_factory=list)


@app.get("/health")
def health() -> dict:
    """Проверка работоспособности сервиса."""
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Возвращает релевантные чанки из векторной базы по текстовому запросу."""
    if not request.query.strip():
        return SearchResponse(chunks=[])

    embedding_service, vector_store = _get_services()
    settings = get_settings()

    top_k = request.top_k or settings.search_top_k
    candidate_limit = max(top_k * settings.search_candidate_multiplier, top_k)

    logger.debug("Поиск по запросу длиной %s символов, top_k=%s.", len(request.query), top_k)

    results = embedding_service.embed_texts([request.query])
    if not results:
        return SearchResponse(chunks=[])

    search_results = vector_store.search(
        results[0].embedding,
        query=request.query,
        limit=top_k,
        candidate_limit=candidate_limit,
        min_score=settings.search_min_score,
        keyword_limit=settings.search_keyword_limit,
    )

    chunks = [item.text for item in search_results]
    response_results = [
        SearchResultResponse(
            text=item.text,
            metadata=item.metadata,
            distance=item.distance,
            score=item.score,
            semantic_score=item.semantic_score,
            keyword_score=item.keyword_score,
            matched_terms=item.matched_terms,
        )
        for item in search_results
    ]

    logger.debug("Найдено чанков: %s.", len(chunks))
    return SearchResponse(chunks=chunks, results=response_results)
