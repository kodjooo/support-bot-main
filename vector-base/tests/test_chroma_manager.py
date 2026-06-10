from collections import defaultdict

from chromadb.errors import NotFoundError

from app.chroma_manager import SearchResult, VectorStoreGateway


class FakeCollection:
    def __init__(self) -> None:
        self.deleted = []
        self.added = []
        self.queries = []

    def delete(self, where):
        self.deleted.append(where)

    def add(self, ids, documents, embeddings, metadatas):
        self.added.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            },
        )

    def query(self, query_embeddings, n_results):
        self.queries.append((query_embeddings, n_results))
        return {
            "documents": [["общий текст", "расшифровка удержаний в дэшборде"]],
            "metadatas": [[{"title": "Склад"}, {"title": "Удержания"}]],
            "distances": [[0.1, 0.3]],
        }

    def get(self, include, limit):
        return {
            "documents": ["детализация самовыкупа показывает сумму удержаний"],
            "metadatas": [{"title": "Самовыкупы"}],
        }


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()
        self.calls = defaultdict(int)
        self.deleted_collections = []

    def get_or_create_collection(self, name):
        self.calls[name] += 1
        return self.collection

    def delete_collection(self, name):
        self.deleted_collections.append(name)


class FlakyCollection(FakeCollection):
    def __init__(self) -> None:
        super().__init__()
        self.fail_once = True

    def query(self, query_embeddings, n_results):
        if self.fail_once:
            self.fail_once = False
            raise NotFoundError("collection disappeared")
        return super().query(query_embeddings, n_results)


class RotatingClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.collections = [FlakyCollection(), FakeCollection()]

    def get_or_create_collection(self, name):
        self.calls[name] += 1
        if len(self.collections) > 1:
            return self.collections.pop(0)
        return self.collections[0]


class StubSettings:
    chroma_host = "chroma"
    chroma_port = 8000
    chroma_collection_name = "knowledge"


def test_replace_document_rewrites_embeddings():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    manager.replace_document(
        doc_id="doc1",
        texts=["a", "b"],
        embeddings=[[0.1], [0.2]],
        metadatas=[{"doc_id": "doc1", "chunk": 0}, {"doc_id": "doc1", "chunk": 1}],
    )

    collection = manager._get_collection()
    assert collection.deleted == [{"doc_id": "doc1"}]
    assert collection.added[0]["ids"] == ["doc1-0", "doc1-1"]


def test_replace_corpus_recreates_collection():
    client = FakeClient()
    manager = VectorStoreGateway(settings=StubSettings(), client=client)

    manager.replace_corpus(
        ids=["chunk-1"],
        texts=["ответ"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"doc_id": "doc-1"}],
    )

    assert client.deleted_collections == ["knowledge"]
    assert client.collection.added[0]["ids"] == ["chunk-1"]
    assert client.collection.added[0]["documents"] == ["ответ"]


def test_query_returns_payload():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.query([0.3, 0.4], limit=2)

    assert "documents" in result
    assert manager._get_collection().queries[0][1] == 2


def test_query_refreshes_stale_collection():
    client = RotatingClient()
    manager = VectorStoreGateway(settings=StubSettings(), client=client)

    result = manager.query([0.3, 0.4], limit=2)

    assert "documents" in result
    assert client.calls["knowledge"] == 2
    assert manager._get_collection().queries[0][1] == 2


def test_search_boosts_exact_terms():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.search(
        [0.3, 0.4],
        query="где расшифровка удержаний",
        limit=1,
        candidate_limit=2,
    )

    assert result[0].text == "расшифровка удержаний в дэшборде"
    assert result[0].matched_terms == ["расшифровк", "удержан"]


def test_search_uses_keyword_candidates_outside_semantic_results():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.search(
        [0.3, 0.4],
        query="детализация самовыкупа удержаний",
        limit=1,
        candidate_limit=1,
    )

    assert result[0].text == "детализация самовыкупа показывает сумму удержаний"


def test_navigation_intent_boosts_actionable_ui_fragments():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    ranked = manager._rerank_results(
        "где посмотреть себестоимость",
        [
            SearchResult(
                text="Себестоимость влияет на расчет прибыли и участвует в финансовых показателях.",
                metadata={},
                semantic_score=0.8,
            ),
            SearchResult(
                text="В разделе «Товары» нажмите на колонку «Себестоимость», откроется окно редактирования.",
                metadata={"section": "Товары", "title": "Себестоимость"},
                semantic_score=0.2,
            ),
        ],
    )

    assert ranked[0].metadata["section"] == "Товары"


def test_rerank_expands_marketplace_and_reconciliation_terms():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    ranked = manager._rerank_results(
        "Почему выплаты WB и Ozon могут не сходиться с продажами?",
        [
            SearchResult(
                text="Для Ozon данные о продажах в страны СНГ не передаются по API.",
                metadata={"section": "Дашборд", "title": "Налоговая база Ozon"},
                semantic_score=0.8,
            ),
            SearchResult(
                text=(
                    "Сумма выплат может не совпадать с кабинетом маркетплейса из-за задержек. "
                    "У Wildberries выплаты происходят с задержкой около четырёх недель, у Ozon задержка может достигать одного месяца."
                ),
                metadata={"section": "Дашборд", "title": "Задержки и особенности расчёта суммы выплат"},
                semantic_score=0.2,
            ),
        ],
    )

    assert ranked[0].metadata["title"] == "Задержки и особенности расчёта суммы выплат"


def test_instruction_intent_boosts_step_fragments():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    ranked = manager._rerank_results(
        "как загрузить таблицу себестоимости",
        [
            SearchResult(
                text="Таблица себестоимости содержит данные о товарах и ценах.",
                metadata={},
                semantic_score=0.8,
            ),
            SearchResult(
                text="Для загрузки таблицы необходимо скачать шаблон, заполнить файл и нажать кнопку «Загрузить таблицу».",
                metadata={"section": "Товары", "title": "Загрузка таблицы"},
                semantic_score=0.2,
            ),
        ],
    )

    assert ranked[0].metadata["title"] == "Загрузка таблицы"


def test_rerank_prefers_early_dense_matches():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    ranked = manager._rerank_results(
        "где посмотреть остатки товара",
        [
            SearchResult(
                text=(
                    "Раздел содержит общую аналитику, продажи, возвраты, периоды, фильтры, "
                    "настройки и в конце упоминает остатки товара."
                ),
                metadata={"section": "Дэшборд"},
                semantic_score=0.8,
            ),
            SearchResult(
                text="Остатки товара отображаются в разделе «Склад», где можно выбрать фильтр по товарам.",
                metadata={"section": "Склад", "title": "Остатки товара"},
                semantic_score=0.2,
            ),
        ],
    )

    assert ranked[0].metadata["section"] == "Склад"


def test_rerank_returns_focused_excerpt():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    ranked = manager._rerank_results(
        "где посмотреть удержания",
        [
            SearchResult(
                text=(
                    "Первое предложение про общую аналитику. "
                    "Второе предложение про настройки. "
                    "В таблице продаж есть сумма удержаний. "
                    "В колонке «Инфо» можно открыть детализацию. "
                    "Последнее предложение про другой раздел."
                ),
                metadata={},
                semantic_score=0.5,
            ),
        ],
    )

    assert "сумма удержаний" in ranked[0].text
    assert "общую аналитику" not in ranked[0].text
