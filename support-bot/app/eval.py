"""Эвал-харнесс качества RAG: прогоняет тест-вопросы через реальный пайплайн
(planner -> /search -> reranker -> ассистент) и считает попадание/качество.

Запуск (внутри контейнера, где доступны прокси OpenAI и vector-base):
    docker compose exec bot-host python -m app.eval                 # уровень planner, все кейсы
    docker compose exec bot-host python -m app.eval --level retrieval
    docker compose exec bot-host python -m app.eval --level full    # + ответ ассистента
    docker compose exec bot-host python -m app.eval --no-images      # пропустить кейсы с картинками
    docker compose exec bot-host python -m app.eval --images-only
    docker compose exec bot-host python -m app.eval --only rashod-edit-location
    docker compose exec bot-host python -m app.eval --tag дашборд

Уровни:
    retrieval — берём query из кейса (или вопрос) как есть -> /search -> rerank.
                Проверяет качество эмбеддингов/поиска без planner.
    planner   — вопрос (+картинка) -> plan_query -> search_query -> /search -> rerank.
                Проверяет, как planner строит запрос (в т.ч. по картинке). [по умолчанию]
    full      — + ответ ассистента. Проверяет конечный текст.

Кейсы — support-bot/eval_cases.json (список объектов). Поля кейса:
    id              уникальный идентификатор
    question        текст вопроса пользователя
    query           (опц.) явный поисковый запрос для уровня retrieval
    image           (опц.) имя файла в eval_images/ (картинка пользователя)
    tag             (опц.) метка для фильтра --tag
    expect_doc      (опц.) подстрока, которая должна встретиться в doc_id отобранного чанка
    expect_section  (опц.) подстрока в section отобранного чанка
    expect_in_answer    (опц.) список подстрок, обязательных в ответе (уровень full)
    forbid_in_answer    (опц.) список подстрок, запрещённых в ответе
    expect_conversational (опц.) true — ждём, что planner вернёт разговорную реплику
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path

from app.config import settings

_BASE = Path(__file__).resolve().parent  # каталог app/ (копируется в образ)
CASES_PATH = _BASE / "eval_cases.json"
IMAGES_DIR = _BASE / "eval_images"


def _image_data_url(name: str) -> str:
    path = IMAGES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Картинка кейса не найдена: {path}")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    ext = path.suffix.lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{b64}"


def _doc_ids(chunks) -> list[str]:
    return [(c.metadata or {}).get("doc_id", "") for c in chunks]


def _hit(chunks, expect_doc: str | None, expect_section: str | None) -> bool:
    for c in chunks:
        meta = c.metadata or {}
        if expect_doc and expect_doc.lower() in (meta.get("doc_id", "") or "").lower():
            return True
        if expect_section and expect_section.lower() in (meta.get("section", "") or "").lower():
            return True
    return False


async def run_case(case: dict, level: str) -> dict:
    from app.ai.planner import plan_query
    from app.ai.vector_client import fetch_context
    from app.ai.reranker import rerank_context
    from app.ai.assistant import call_assistant

    question = case.get("question", "")
    image_urls = [_image_data_url(case["image"])] if case.get("image") else []
    res: dict = {"id": case["id"], "checks": [], "ok": True}

    # 1) Поисковый запрос
    if level == "retrieval":
        search_query = case.get("query") or question
        planning = None
    else:
        planning = await plan_query([question], image_urls=image_urls)
        res["planner_status"] = planning.status
        res["search_query"] = planning.search_query
        if case.get("expect_conversational"):
            ok = planning.is_conversational
            res["checks"].append(("conversational", ok))
            res["ok"] = ok
            return res
        search_query = planning.search_query or question

    # 2) Поиск
    chunks = await fetch_context(search_query, top_k=settings.rag_search_top_k)
    res["retrieved"] = _doc_ids(chunks)[:settings.rag_search_top_k]
    res["retrieval_hit"] = _hit(chunks, case.get("expect_doc"), case.get("expect_section"))

    # 3) Reranker
    selected = []
    if chunks:
        rer = await rerank_context(user_query=question, search_query=search_query, chunks=chunks)
        selected = [chunks[i] for i in rer.selected_indices if 0 <= i < len(chunks)]
        res["kept"] = _doc_ids(selected)
        res["enough_context"] = rer.enough_context
    is_gap = bool(case.get("expect_gap"))
    kept_hit = _hit(selected, case.get("expect_doc"), case.get("expect_section"))
    res["kept_hit"] = kept_hit
    if not is_gap and (case.get("expect_doc") or case.get("expect_section")):
        res["checks"].append(("kept_hit", kept_hit))

    # 4) Ответ ассистента (уровень full)
    if level == "full":
        prefix = "Контекст из базы знаний (отобран RAG):\n\n" + "\n\n".join(
            c.to_prompt_text() for c in selected
        )
        answer, needs_op, _ = await call_assistant(None, [prefix, question], image_urls)
        res["answer"] = answer
        res["needs_operator"] = needs_op
        text = (answer or "").lower()
        for sub in case.get("expect_in_answer", []):
            res["checks"].append((f"includes:{sub}", sub.lower() in text))
        for sub in case.get("forbid_in_answer", []):
            res["checks"].append((f"forbids:{sub}", sub.lower() not in text))
        if is_gap:
            # Пробел: ок, если бот не выдумал, а корректно отказался / перевёл на поддержку.
            decline = ("поддержк", "оператор", "уточн", "не могу", "пока не", "в планах",
                       "не предусмотрен", "свяжитесь", "напишите", "не реализован", "только wildberries")
            gap_ok = needs_op or any(w in text for w in decline)
            res["checks"].append(("graceful_gap", gap_ok))

    if is_gap and level != "full":
        res["ok"] = True  # на уровне planner/retrieval пробелы только наблюдаем
    else:
        res["ok"] = all(ok for _, ok in res["checks"]) if res["checks"] else res.get("kept_hit", True)
    return res


def _filter(cases: list[dict], args) -> list[dict]:
    out = []
    for c in cases:
        if args.only and c["id"] != args.only:
            continue
        if args.tag and args.tag not in (c.get("tag", "") or ""):
            continue
        if args.no_images and c.get("image"):
            continue
        if args.images_only and not c.get("image"):
            continue
        out.append(c)
    return out


async def main_async(args) -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    cases = _filter(cases, args)
    if not cases:
        print("Нет кейсов под фильтр.")
        return 0
    passed = 0
    for case in cases:
        try:
            res = await run_case(case, args.level)
        except Exception as e:  # один кейс не должен валить прогон
            print(f"[ОШИБКА] {case['id']}: {e}")
            continue
        mark = "PASS" if res["ok"] else "FAIL"
        if res["ok"]:
            passed += 1
        print(f"\n[{mark}] {res['id']}")
        if "search_query" in res:
            print(f"  planner={res.get('planner_status')} query={res.get('search_query')!r}")
        if "retrieved" in res:
            print(f"  retrieved={res['retrieved']}")
        if "kept" in res:
            print(f"  kept={res['kept']} enough={res.get('enough_context')}")
        if "answer" in res:
            print(f"  answer={(res['answer'] or '')[:300]}")
        for name, ok in res["checks"]:
            print(f"    {'✓' if ok else '✗'} {name}")
    print(f"\n==== ИТОГ: {passed}/{len(cases)} PASS (level={args.level}) ====")
    return 0 if passed == len(cases) else 1


def main() -> None:
    p = argparse.ArgumentParser(description="Эвал качества RAG карты интерфейса и базы.")
    p.add_argument("--level", choices=["retrieval", "planner", "full"], default="planner")
    p.add_argument("--only", help="ID одного кейса")
    p.add_argument("--tag", help="Фильтр по метке")
    p.add_argument("--no-images", action="store_true", help="Пропустить кейсы с картинками")
    p.add_argument("--images-only", action="store_true", help="Только кейсы с картинками")
    args = p.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
