"""Генератор карты интерфейса: скриншоты -> текстовые описания -> RAG-чанки.

Два этапа (между ними — ручная выверка sources/*.json):

  recognize  скриншоты screens/*.png -> vision-модель -> sources/<stem>.json
  build      выверенные sources/*.json -> ui_map_chunks.jsonl (схема корпуса)

Имя скрина: <раздел>__<подраздел>[__состояние][@<маркетплейс>].png. Из ключа
выводятся doc_id и section_path. На этапе build одинаковые WB/Ozon-описания
схлопываются в один чанк marketplace=common, различающиеся — в отдельные чанки.

См. docs/plans/ui-map.md.

Запуск (из vector-base/):
    python -m artifacts.ui_map.build_ui_map recognize [--force] [--only <stem>]
    # (выверка sources/*.json вручную)
    python -m artifacts.ui_map.build_ui_map build

Зависимости (офлайн-инструмент): openai (есть в проекте). Vision-модель —
OPENAI_VISION_MODEL.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Импорты app.config/app.openai_client нужны только для recognize (vision через
# OpenAI) и вынесены внутрь функции, чтобы команда build работала на чистом
# stdlib без зависимостей и без сетевого доступа.

logger = logging.getLogger("build_ui_map")

HERE = Path(__file__).resolve().parent
SCREENS_DIR = HERE / "screens"
SOURCES_DIR = HERE / "sources"
MANIFEST_PATH = HERE / "manifest.json"
CHUNKS_JSONL = HERE / "ui_map_chunks.jsonl"
CHUNKS_JSON = HERE / "ui_map_chunks.json"
CUSTOM_FACTS = HERE / "custom_facts.jsonl"  # ручные текстовые факты (не из скринов)

# Подписи маркетплейсов для человекочитаемых текстов.
MARKETPLACE_TITLES = {"wb": "Wildberries", "ozon": "Ozon", "ym": "Yandex Market", "common": "общий вид"}

# Экраны, маркетплейс-агностичные по сути (модалки/формы с одинаковой структурой
# на WB и Ozon, без МП-специфичных данных) + рекламные копии (Ozon-кадры — копии
# WB, т.к. данных Ozon-рекламы нет, отличие минимально). Для них держим ОДИН набор
# чанков marketplace=common вместо дублей по МП. Списки/таблицы сюда НЕ входят —
# там данные различаются по маркетплейсу. Проверено перцептивным хэшем (dHash≤18).
COMMON_KEYS = {
    "реклама__товары", "реклама__бренды", "реклама__товары-раскрыто",
    "реклама__фильтр-типы", "реклама__фильтр-статусы",
    "реклама__товары-поиск рекламной кампании",
    "реклама__товары-расшифровка количества просмотров",
    "тариф__список", "тариф__модалка-купить",
    "подключение__магазин",
    "расходы__модалка-создание", "расходы__модалка-редактирование", "расходы__модалка-фильтр",
    "настройки__модалка-настройки", "настройки__модалка-удалить",
    "дашборд__модалка-период", "дашборд__плитка-детализация", "дашборд__строка-детализация",
    "дашборд__плитки__заказы__артикул-детализация", "дашборд__плитки__заказы__заказам-детализация",
}
_COMMON_KEYS_NFC = {unicodedata.normalize("NFC", k) for k in COMMON_KEYS}

# Тривиальная «мебель» интерфейса — пропускаем при сборке чанков (шум для поиска).
_ELEMENT_DENY = (
    "информационный баннер", "баннер-анонс", "баннер ", "анонс инструмента",
    "заголовок страницы", "заголовок модал", "заголовок окна", "заголовок формы",
    "боковое меню", "сайдбар", "навигация по разделам", "левое меню",
    "столбцы таблицы", "шапка таблицы", "колонки таблицы", "заголовки колонок",
    "заголовки столбцов", "заголовки метрик",
    "пагинация", "кнопка закрытия", "крестик закрытия", "иконка закрытия",
    "верхняя панель навигации", "хлебные крошки",
)

VISION_PROMPT = (
    "Ты документируешь интерфейс веб-сервиса аналитики для продавцов маркетплейсов. "
    "На изображении — один экран (или модальное окно) сервиса. Опиши СТРОГО то, что видно, "
    "не выдумывай функциональность. Верни ТОЛЬКО JSON-объект по схеме:\n"
    "{\n"
    '  "screen_title": "человекочитаемое название экрана",\n'
    '  "ui_state": "screen | overlay | modal",\n'
    '  "screen_summary": "1-3 предложения: что это за экран и зачем он",\n'
    '  "elements": [\n'
    "    {\n"
    '      "name": "название элемента (кнопка/поле/вкладка/иконка/колонка)",\n'
    '      "location": "где находится, словами пользователя (вверху справа, в колонке ..., в строке товара)",\n'
    '      "purpose": "что делает / что показывает",\n'
    '      "appearance": "как выглядит (иконка, цвет, подпись)",\n'
    '      "aliases": ["как пользователь может это назвать другими словами"]\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "Расположение пиши словами пользователя. Иконки/цвета называй так, как их назвал бы пользователь "
    "(«восклицательный знак», «красная плашка»). Подписи и тексты — на русском, как на экране."
)


# --------------------------- разбор имени экрана ---------------------------

def parse_stem(stem: str) -> tuple[str, str | None]:
    """'товары__список@wb' -> ('товары__список', 'wb'); без @ -> (stem, None)."""
    if "@" in stem:
        base, mp = stem.rsplit("@", 1)
        return base, mp
    return stem, None


def doc_id_for(base_key: str, marketplace: str) -> str:
    """Базовый ключ + маркетплейс -> стабильный doc_id.

    common -> без суффикса маркетплейса; конкретный маркетплейс -> с суффиксом."""
    segments = base_key.split("__")
    doc_id = "ui:" + ":".join(segments)
    if marketplace != "common":
        doc_id += f":{marketplace}"
    return doc_id


def section_path_for(base_key: str) -> list[str]:
    """Сегменты ключа -> человекочитаемый section_path (первая буква заглавная)."""
    return [seg[:1].upper() + seg[1:] for seg in base_key.split("__")]


# --------------------------- этап recognize ---------------------------

def _read_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {"format": "sellerdata_ui_map_v1", "screens": {}}


def _png_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _strip_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def recognize(only: str | None, force: bool) -> int:
    # Импорт здесь, а не на уровне модуля: нужен только для vision через OpenAI.
    from app.config import get_settings
    from app.openai_client import create_openai_client

    settings = get_settings()
    client = create_openai_client(settings, timeout=180.0, max_retries=2)
    manifest = _read_manifest()
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    pngs = sorted(SCREENS_DIR.glob("*.png"))
    if only:
        pngs = [p for p in pngs if p.stem == only]
    processed = 0
    for png in pngs:
        stem = png.stem
        png_hash = _png_hash(png)
        source_path = SOURCES_DIR / f"{stem}.json"
        recorded = manifest["screens"].get(stem, {})
        if not force and recorded.get("png_sha256") == png_hash and source_path.exists():
            logger.info("Без изменений, пропуск: %s", stem)
            continue

        base_key, marketplace = parse_stem(stem)
        logger.info("Распознаю %s (vision)...", stem)
        data_url = "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii")
        response = client.chat.completions.create(
            model=settings.openai_vision_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                }
            ],
        )
        parsed = _strip_json(response.choices[0].message.content or "{}")
        parsed["screen_key"] = base_key
        parsed["marketplace"] = marketplace or "common"
        source_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

        manifest["screens"][stem] = {
            "base_key": base_key,
            "marketplace": marketplace or "common",
            "doc_id": doc_id_for(base_key, marketplace or "common"),
            "section_path": section_path_for(base_key),
            "png_sha256": png_hash,
            "recognized_at": datetime.now(timezone.utc).isoformat(),
        }
        processed += 1
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Распознано/обновлено: %s. Дальше выверьте sources/*.json и запустите build.", processed)
    return processed


# --------------------------- этап build ---------------------------

def _content_signature(source: dict[str, Any]) -> str:
    """Нормализованная подпись содержимого экрана для дедупа WB/Ozon."""
    parts = [source.get("screen_summary", "")]
    for el in source.get("elements", []):
        parts.append(f"{el.get('name','')}|{el.get('location','')}|{el.get('purpose','')}")
    text = re.sub(r"\s+", " ", " ".join(parts)).strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_chunks(base_key: str, marketplace: str, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Описание экрана -> чанки (обзор + по элементу) в схеме корпуса."""
    doc_id = doc_id_for(base_key, marketplace)
    section_path = section_path_for(base_key)
    section = " > ".join(section_path)
    screen_title = source.get("screen_title") or section
    ui_state = source.get("ui_state") or "screen"
    mp_title = MARKETPLACE_TITLES.get(marketplace, marketplace)
    mp_hint = "" if marketplace == "common" else f" (маркетплейс: {mp_title})"

    chunks: list[dict[str, Any]] = []

    def emit(index: int, title: str, answer: str, intents: list[str], keywords: list[str], facts: list[str]) -> None:
        embedding_text = "\n".join(
            part for part in [
                f"Раздел: {section}",
                f"Экран: {screen_title}{mp_hint}",
                f"Заголовок: {title}",
                f"Вопросы: {'; '.join(intents)}" if intents else "",
                f"Ключевые слова: {', '.join(keywords)}" if keywords else "",
                f"Ответ: {answer}",
            ] if part
        )
        content_hash = hashlib.sha256(embedding_text.encode("utf-8")).hexdigest()[:16]
        chunks.append({
            "id": f"{doc_id}:{index:04d}:{content_hash}",
            "doc_id": doc_id,
            "doc_title": screen_title,
            "doc_modified_time": "",
            "chunk_index": index,
            "section_path": section_path,
            "block_type": "ui_location",
            "title": title,
            "question_intents": intents,
            "keywords": keywords,
            "answer": answer,
            "source_facts": facts,
            "embedding_text": embedding_text,
            "metadata": {
                "doc_id": doc_id,
                "doc_title": screen_title,
                "section": section,
                "title": title,
                "block_type": "ui_location",
                "ui_state": ui_state,
                "parent_screen": source.get("parent_screen") or "",
                "marketplace": marketplace,
                "keywords": ", ".join(keywords[:20]),
                "preview": answer[:240],
            },
        })

    # Обзорный чанк экрана.
    summary = (source.get("screen_summary") or "").strip()
    if summary:
        emit(
            0,
            f"Обзор экрана «{screen_title}»{mp_hint}",
            summary,  # summary от агента уже естественный — без технических меток
            [f"что показывает {screen_title}", f"для чего раздел {section}", f"где находится {screen_title}"],
            section_path,
            [summary],
        )

    # Чанк на каждый элемент (кроме тривиальной «мебели» интерфейса — она не несёт
    # полезного «где/зачем», но общими словами «назначение/баннер/заголовок»
    # цепляется к запросам и вытесняет правильный фрагмент).
    for i, el in enumerate(source.get("elements", []) or [], start=1):
        name = (el.get("name") or "").strip()
        if not name:
            continue
        name_low = name.lower()
        if any(stop in name_low for stop in _ELEMENT_DENY):
            continue
        location = (el.get("location") or "").strip()
        purpose = (el.get("purpose") or "").strip()
        appearance = (el.get("appearance") or "").strip()
        aliases = [a.strip() for a in (el.get("aliases") or []) if a.strip()]
        # Человекочитаемый ответ: расположение + назначение. Без технических меток
        # (раздел/экран/маркетплейс хранятся в embedding_text/metadata для поиска).
        answer = f"«{name}» — {location}." if location else f"«{name}»."
        if purpose:
            answer += f" {purpose}"
        intents = [f"где находится {name}", f"что такое {name}", f"как {purpose[:40]}" if purpose else ""]
        intents = [s for s in intents if s] + aliases
        keywords = list(dict.fromkeys([name, *aliases, *section_path]))
        facts = [f for f in [f"{name}: {location}", purpose, appearance] if f]
        emit(i, f"Где находится «{name}»", answer, intents, keywords, facts)

    return chunks


def build() -> int:
    sources: dict[str, dict[str, dict[str, Any]]] = {}  # base_key -> {marketplace -> source}
    for path in sorted(SOURCES_DIR.glob("*.json")):
        base_key, marketplace = parse_stem(path.stem)
        marketplace = marketplace or "common"
        source = json.loads(path.read_text(encoding="utf-8"))
        sources.setdefault(base_key, {})[marketplace] = source

    all_chunks: list[dict[str, Any]] = []
    common_count = mp_specific_count = 0
    for base_key, by_mp in sources.items():
        # Курируемый список маркетплейс-агностичных экранов → один общий набор.
        # Нормализуем Unicode (имена файлов бывают в NFD, ключи — в NFC).
        if unicodedata.normalize("NFC", base_key) in _COMMON_KEYS_NFC:
            chosen = by_mp.get("wb") or next(iter(by_mp.values()))
            all_chunks.extend(_build_chunks(base_key, "common", chosen))
            common_count += 1
            logger.info("common (курировано): %s (%s)", base_key, ", ".join(by_mp))
            continue
        # Дедуп WB/Ozon: если описания идентичны — один чанк-набор marketplace=common.
        signatures = {mp: _content_signature(src) for mp, src in by_mp.items()}
        non_common = {mp: sig for mp, sig in signatures.items() if mp != "common"}
        if "common" in by_mp:
            # Уже единый источник.
            all_chunks.extend(_build_chunks(base_key, "common", by_mp["common"]))
            common_count += 1
        elif len(non_common) > 1 and len(set(non_common.values())) == 1:
            # Все маркетплейсы дали одинаковое описание -> общий чанк.
            any_mp = next(iter(by_mp))
            all_chunks.extend(_build_chunks(base_key, "common", by_mp[any_mp]))
            common_count += 1
            logger.info("Схлопнут в common: %s (%s)", base_key, ", ".join(by_mp))
        else:
            # Описания различаются -> отдельные чанки на маркетплейс.
            for mp, src in by_mp.items():
                all_chunks.extend(_build_chunks(base_key, mp, src))
                mp_specific_count += 1

    # Домержив ручные текстовые факты (тариф, группировки, реклама, тултипы и т.п.).
    custom_count = 0
    if CUSTOM_FACTS.exists():
        for line in CUSTOM_FACTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                fact = json.loads(line)
                if not (fact.get("id") and fact.get("embedding_text") and fact.get("answer")):
                    raise ValueError(f"custom_facts: в записи нет id/embedding_text/answer: {line[:80]}")
                all_chunks.append(fact)
                custom_count += 1
        logger.info("Добавлено ручных фактов: %s", custom_count)

    CHUNKS_JSONL.write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in all_chunks), encoding="utf-8"
    )
    CHUNKS_JSON.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "Собрано чанков: %s (экранов common: %s, маркетплейс-специфичных групп: %s). Файл: %s",
        len(all_chunks), common_count, mp_specific_count, CHUNKS_JSONL,
    )
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор карты интерфейса (vision -> JSONL).")
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("recognize", help="Распознать скрины в sources/*.json (vision).")
    rec.add_argument("--only", help="Только один скрин по stem (имя файла без .png).")
    rec.add_argument("--force", action="store_true", help="Перераспознать даже неизменённые.")
    sub.add_parser("build", help="Собрать ui_map_chunks.jsonl из выверенных sources/*.json.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "recognize":
        recognize(args.only, args.force)
    else:
        build()


if __name__ == "__main__":
    main()
