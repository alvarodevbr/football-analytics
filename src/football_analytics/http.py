"""Cliente HTTP educado para o Transfermarkt: retry com backoff, jitter e detecção de bloqueio."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .logging_utils import get_logger

log = get_logger("http")


class FetchError(RuntimeError):
    """Falha definitiva ao obter um recurso (após os retries)."""


class BlockedError(FetchError):
    """O servidor está recusando o cliente (403/429 persistentes): parar a coleta, não insistir."""


@dataclass
class Throttle:
    """Intervalo mínimo entre requisições, com jitter uniforme para não gerar padrão fixo."""

    delay: float
    jitter: float = 0.0
    _last: float = 0.0

    def wait(self) -> None:
        target = self._last + self.delay + random.uniform(0, self.jitter)
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        self._last = time.monotonic()


def make_session(user_agent: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Language": "en-US,en;q=0.9",
    })
    # O adapter só refaz falhas de conexão; status HTTP (5xx, 403, 429) são tratados em `get`,
    # com backoff longo e exceções próprias do pacote.
    retry = Retry(total=2, connect=2, read=0, status=0, backoff_factor=1.0, allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def get(
    session: requests.Session,
    url: str,
    throttle: Throttle,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 45.0,
    attempts: int = 5,
    backoff_seconds: float = 15.0,
) -> requests.Response:
    """GET com throttle e backoff exponencial em 5xx, 403/429 e falhas de transporte.

    Levanta `BlockedError` se 403/429 persistirem em todas as tentativas e `FetchError` para
    qualquer outra falha definitiva (inclusive 404, que não é re-tentado).
    """
    last_error: str = ""
    blocked = False
    for attempt in range(1, attempts + 1):
        throttle.wait()
        try:
            resp = session.get(url, headers=headers, timeout=timeout)
        except requests.RequestException as exc:  # timeout, conexão, RetryError do adapter...
            last_error, blocked = f"{type(exc).__name__}: {exc}", False
        else:
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                raise FetchError(f"404 em {url}")
            # Desafio de captcha do AWS WAF (o Transfermarkt responde 405 + este cabeçalho): não se
            # resolve com espera curta, então para a coleta imediatamente.
            if resp.headers.get("x-amzn-waf-action"):
                raise BlockedError(f"{url}: desafio WAF ({resp.headers['x-amzn-waf-action']}, "
                                   f"HTTP {resp.status_code}) — interrompa e retome mais tarde")
            blocked = resp.status_code in (403, 429)
            last_error = f"HTTP {resp.status_code}"
        wait = backoff_seconds * 2 ** (attempt - 1)
        log.warning("requisição falhou; nova tentativa",
                    extra={"url": url, "attempt": attempt, "error": last_error, "wait_s": wait})
        if attempt < attempts:
            time.sleep(wait)
    if blocked:
        raise BlockedError(f"{url}: {last_error} persistente — possível bloqueio; interrompa e retome depois")
    raise FetchError(f"{url}: {last_error}")
