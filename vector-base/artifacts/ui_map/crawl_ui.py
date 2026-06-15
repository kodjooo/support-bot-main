"""Кравлер интерфейса Sellerdata для автоматического снятия скриншотов.

Читает crawl_map.yaml, логинится под демо-аккаунтом, для каждого маркетплейса
переключает активный магазин и обходит все экраны из списка screens, выполняя
их actions (открыть вкладку/модалку/состояние), затем снимает скриншот в
screens/<key>@<mp>.png.

Дальше эти PNG идут в build_ui_map (vision-распознавание -> sources -> JSONL).
См. docs/plans/ui-map.md.

Запуск (из vector-base/):
    python -m artifacts.ui_map.crawl_ui [--only дашборд__график] \
        [--marketplace wb] [--headed] [--full-page]

Доступы и адрес стенда берутся ТОЛЬКО из переменных окружения, в crawl_map.yaml
не хранятся:
    SELLERDATA_BASE_URL          адрес тестового стенда
    SELLERDATA_DEMO_EMAIL        e-mail демо-аккаунта
    SELLERDATA_DEMO_PASSWORD     пароль демо-аккаунта
    SELLERDATA_WB_STORE_TITLE    подпись WB-магазина в выпадашке
    SELLERDATA_OZON_STORE_TITLE  подпись Ozon-магазина в выпадашке

Зависимости (офлайн-инструмент, не runtime): playwright, pyyaml.
    pip install playwright pyyaml && playwright install chromium
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from playwright.sync_api import (
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

logger = logging.getLogger("crawl_ui")

HERE = Path(__file__).resolve().parent
DEFAULT_MAP = HERE / "crawl_map.yaml"
SCREENS_DIR = HERE / "screens"

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _resolve_env(value: Any) -> Any:
    """Подставляет ${VAR} из окружения в строках (рекурсивно по dict/list)."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise RuntimeError(f"Не задана переменная окружения {name}.")
            return resolved

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


def _load_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or "screens" not in raw:
        raise ValueError(f"Некорректный crawl_map: {path}")
    return raw


def _safe_goto(page: Page, url: str, timeout: int = 30000, attempts: int = 3) -> None:
    """goto с ретраем на ERR_ABORTED: SPA-навигации Inertia вытесняют друг друга,
    из-за чего прямой переход иногда прерывается. Это безобидно — повторяем."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            return
        except PlaywrightError as error:
            if "ERR_ABORTED" not in str(error):
                raise
            last_error = error
            page.wait_for_timeout(500 * (attempt + 1))
    if last_error:
        raise last_error


def _wait(page: Page, spec: dict[str, Any] | None, default_timeout: int) -> None:
    """Ожидание по спецификации wait_for: {hidden|visible|url, timeout_ms}.

    hidden принимает строку или список селекторов: ждём, пока КАЖДЫЙ из них не
    исчезнет. Отсутствующий на странице селектор условие 'hidden' удовлетворяет
    мгновенно, поэтому можно перечислять все возможные лоадеры разом."""
    if not spec:
        return
    timeout = int(spec.get("timeout_ms", default_timeout))
    if "hidden" in spec:
        targets = spec["hidden"] if isinstance(spec["hidden"], list) else [spec["hidden"]]
        for selector in targets:
            try:
                page.wait_for_selector(selector, state="hidden", timeout=timeout)
            except PlaywrightTimeoutError:
                # Лоадер не исчез за отведённое время — логируем, но не валим обход.
                logger.warning("Лоадер %s не исчез за %sмс.", selector, timeout)
    if "visible" in spec:
        try:
            page.wait_for_selector(spec["visible"], state="visible", timeout=timeout)
        except PlaywrightTimeoutError:
            # Элемент не появился (напр. реклама без данных) — не валим экран.
            logger.warning("Элемент %s не появился за %sмс.", spec["visible"], timeout)
    if "url" in spec:
        page.wait_for_url(spec["url"], timeout=timeout)


def _run_action(page: Page, action: dict[str, Any], default_timeout: int) -> None:
    """Выполняет один шаг сценария: click / fill / click_text / select2 / wait_for."""
    if "click" in action:
        page.click(action["click"], timeout=default_timeout)
    elif "hover" in action:
        # Наведение для показа тултипа (v-tippy). После hover нужна пауза на появление.
        page.hover(action["hover"], timeout=default_timeout)
        page.wait_for_timeout(700)
    elif "click_all" in action:
        # Кликнуть ВСЕ совпадения (раскрыть все аккордеоны модалки и т.п.).
        elements = page.locator(action["click_all"])
        for i in range(elements.count()):
            try:
                elements.nth(i).click(timeout=default_timeout)
            except PlaywrightError:
                pass  # часть может быть уже раскрыта/перекрыта — не критично
    elif "fill" in action:
        page.fill(action["fill"], action["value"], timeout=default_timeout)
    elif "click_text" in action:
        # Клик по видимому тексту (для пунктов без устойчивого селектора).
        page.get_by_text(action["click_text"], exact=False).first.click(timeout=default_timeout)
    elif "select2" in action:
        # Select2: открыть выпадашку кликом по .select2-selection внутри контейнера,
        # затем выбрать опцию из выпавшего списка по тексту.
        page.locator(action["select2"]).locator(".select2-selection").first.click(timeout=default_timeout)
        option = action["option"]
        page.locator("li.select2-results__option", has_text=option).first.click(timeout=default_timeout)
    elif "wait_for" in action:
        _wait(page, action["wait_for"], default_timeout)
    elif "wait_for_url" in action:
        page.wait_for_url(action["wait_for_url"], timeout=default_timeout)
    else:
        raise ValueError(f"Неизвестный тип action: {action}")


def _login(page: Page, base_url: str, login: dict[str, Any], timeout: int) -> None:
    # networkidle не годится: приложение держит постоянный websocket (pusher),
    # сеть никогда не затихает. Ждём загрузку DOM, дальше — явные ожидания.
    _safe_goto(page, base_url.rstrip("/") + login["path"])
    for step in login.get("steps", []):
        _run_action(page, step, timeout)
    logger.info("Вход выполнен.")


def _switch_store(page: Page, base_url: str, switch: dict[str, Any], store_title: str, timeout: int) -> None:
    """Переключает активный магазин (маркетплейс) через выпадашку в шапке."""
    if not switch:
        return
    # Уйти на чистый дашборд: сбрасывает любую открытую модалку/оверлей от
    # предыдущего экрана, иначе backdrop перекрывает выпадашку магазинов.
    _safe_goto(page, base_url.rstrip("/") + "/dashboard")
    page.wait_for_selector(".user_account_status", state="visible", timeout=timeout)
    for step in switch.get("steps", []):
        # Подставляем подпись магазина в шаблон {store_title}.
        resolved = {
            key: (value.replace("{store_title}", store_title) if isinstance(value, str) else value)
            for key, value in step.items()
        }
        _run_action(page, resolved, timeout)
    # changeAccount запускает перезагрузку — дать ей завершиться до следующего goto.
    page.wait_for_timeout(1500)
    logger.info("Активный магазин переключён на «%s».", store_title)


def _screen_filename(key: str, marketplace_id: str | None) -> str:
    return f"{key}@{marketplace_id}.png" if marketplace_id else f"{key}.png"


def _capture_screen(
    page: Page,
    base_url: str,
    screen: dict[str, Any],
    default_wait: dict[str, Any],
    full_page: bool,
    marketplace_id: str | None,
) -> Path:
    _safe_goto(page, base_url.rstrip("/") + screen["path"])
    _wait(page, default_wait, int(default_wait.get("timeout_ms", 15000)))
    timeout = int(default_wait.get("timeout_ms", 15000))
    for action in screen.get("actions", []) or []:
        _run_action(page, action, timeout)
    # Дать догрузиться отложенным элементам (сайдбар, аватар) и завершиться
    # анимациям модалок/дропдаунов, чтобы кадр не снялся раньше полной отрисовки.
    page.wait_for_timeout(1200)
    # Повторно убедиться, что контентные лоадеры исчезли уже после действий.
    _wait(page, default_wait, int(default_wait.get("timeout_ms", 15000)))
    SCREENS_DIR.mkdir(parents=True, exist_ok=True)
    out = SCREENS_DIR / _screen_filename(screen["key"], marketplace_id)
    # full_page можно переопределить на экране (модалки удобнее снимать в viewport).
    screen_full_page = screen.get("full_page", full_page)
    page.screenshot(path=str(out), full_page=screen_full_page)
    return out


def crawl(
    map_path: Path,
    only: str | None,
    marketplace_filter: str | None,
    headed: bool,
    full_page: bool,
) -> list[Path]:
    config = _resolve_env(_load_map(map_path))
    base_url = config["base_url"]
    viewport = config.get("viewport", {"width": 1440, "height": 900})
    default_wait = config.get("default_wait", {})
    # full_page: приоритет у CLI (--viewport-only), иначе из конфига (по умолчанию вся страница).
    if full_page is None:
        full_page = bool(config.get("full_page", True))
    marketplaces = config.get("marketplaces") or [None]
    switch = config.get("switch_store", {})
    screens = config["screens"]
    if only:
        screens = [s for s in screens if s["key"] == only]
        if not screens:
            raise SystemExit(f"Экран с key={only!r} не найден в карте.")

    captured: list[Path] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(viewport=viewport)
        page = context.new_page()
        login_timeout = int(default_wait.get("timeout_ms", 15000))
        _login(page, base_url, config["login"], login_timeout)

        for mp in marketplaces:
            mp_id = mp["id"] if isinstance(mp, dict) else None
            if marketplace_filter and mp_id != marketplace_filter:
                continue
            if isinstance(mp, dict):
                try:
                    _switch_store(page, base_url, switch, mp["store_title"], login_timeout)
                except PlaywrightError as error:
                    logger.warning("Не удалось переключиться на «%s», маркетплейс пропущен: %s",
                                   mp["store_title"], error)
                    continue
            for screen in screens:
                # Экран можно ограничить списком маркетплейсов (напр. Ozon-only страницы).
                allowed = screen.get("marketplaces")
                if allowed and mp_id is not None and mp_id not in allowed:
                    continue
                try:
                    out = _capture_screen(page, base_url, screen, default_wait, full_page, mp_id)
                    captured.append(out)
                    logger.info("Снят %s", out.name)
                except PlaywrightError as error:
                    # Ловим ЛЮБУЮ ошибку Playwright (таймаут, ERR_ABORTED, strict и т.п.),
                    # чтобы один хрупкий экран не ронял весь обход. Логируем и идём дальше.
                    first_line = str(error).split("\n", 1)[0]
                    logger.warning("Пропущен %s@%s: %s", screen["key"], mp_id, first_line)
        context.close()
        browser.close()
    return captured


def main() -> None:
    parser = argparse.ArgumentParser(description="Снимает скриншоты интерфейса по crawl_map.yaml.")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="Путь к crawl_map.yaml.")
    parser.add_argument("--only", help="Снять только один экран по его key.")
    parser.add_argument("--marketplace", help="Снять только один маркетплейс (id из marketplaces).")
    parser.add_argument("--headed", action="store_true", help="Показывать браузер (отладка).")
    parser.add_argument(
        "--viewport-only",
        dest="full_page",
        action="store_false",
        default=None,
        help="Снимать только видимую область, а не всю страницу (по умолчанию — вся страница).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    captured = crawl(args.map, args.only, args.marketplace, args.headed, args.full_page)
    logger.info("Готово. Снято скринов: %s. Каталог: %s", len(captured), SCREENS_DIR)


if __name__ == "__main__":
    main()
