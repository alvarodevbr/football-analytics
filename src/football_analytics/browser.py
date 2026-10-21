"""Chrome controlado pelo seleniumbase (modo UC) para fontes protegidas por anti-bot.

FBref (Cloudflare) e Sofascore recusam clientes HTTP simples (403) e também o Chrome headless.
O soccerdata já encapsula o caso do FBref; o Sofascore usa `BrowserJsonClient`, que executa
``fetch()`` dentro da página, herdando cookies e fingerprint do navegador real.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import TracebackType
from typing import Any

from .http import BlockedError, FetchError, Throttle
from .logging_utils import get_logger

log = get_logger("browser")

_SYSTEM_CHROMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "chrome")

_FETCH_JS = """
const done = arguments[arguments.length - 1];
fetch(arguments[0], {credentials: 'include', headers: {'Accept': 'application/json'}})
  .then(r => r.text().then(t => done([r.status, t])))
  .catch(e => done([0, String(e)]));
"""


def find_chrome() -> str | None:
    """Caminho (str) de um Chrome utilizável, ou None se houver um Chrome de sistema.

    O seleniumbase chama ``.lower()`` no caminho, portanto ele precisa ser `str`, não `Path`.
    Sem Chrome no sistema, usa o "Chrome for Testing" baixado por ``sbase get cft``.
    """
    if any(shutil.which(name) for name in _SYSTEM_CHROMES):
        return None
    import seleniumbase

    drivers = Path(seleniumbase.__file__).parent / "drivers" / "cft_drivers"
    for candidate in sorted(drivers.glob("**/chrome")):
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError("Chrome não encontrado. Instale com:  .venv/bin/sbase get cft")


class BrowserJsonClient:
    """Sessão de navegador que lê endpoints JSON de um site a partir da própria página."""

    def __init__(self, home_url: str, throttle: Throttle, headless: bool = False) -> None:
        self.home_url = home_url
        self.throttle = throttle
        self.headless = headless
        self._driver: Any = None

    def __enter__(self) -> "BrowserJsonClient":
        from seleniumbase import Driver

        self._driver = Driver(uc=True, headless=self.headless, binary_location=find_chrome())
        self._driver.set_script_timeout(60)
        self._driver.uc_open_with_reconnect(self.home_url, 4)
        log.info("navegador iniciado", extra={"home": self.home_url})
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: TracebackType | None) -> None:
        if self._driver is not None:
            self._driver.quit()
            self._driver = None

    def get_json(self, url: str, attempts: int = 3) -> Any:
        if self._driver is None:
            raise RuntimeError("BrowserJsonClient usado fora do bloco `with`")
        last = ""
        for attempt in range(1, attempts + 1):
            self.throttle.wait()
            status, body = self._driver.execute_async_script(_FETCH_JS, url)
            if status == 200:
                return json.loads(body)
            if status == 404:
                raise FetchError(f"404 em {url}")
            last = f"HTTP {status}: {body[:120]}"
            log.warning("fetch no navegador falhou", extra={"url": url, "attempt": attempt, "error": last})
            if status in (403, 429) and attempt < attempts:
                # recarregar a home renova os cookies do desafio anti-bot
                self._driver.uc_open_with_reconnect(self.home_url, 6)
        if last.startswith(("HTTP 403", "HTTP 429")):
            raise BlockedError(f"{url}: {last}")
        raise FetchError(f"{url}: {last}")
