"""Transfermarkt: clubes, elencos por temporada, histórico de valores, contrato e transferências.

Acesso por `requests` (sem Cloudflare). Duas classes de recurso:

* páginas HTML (liga, elenco, perfil), parseadas com lxml;
* endpoints JSON públicos usados pelo próprio site (``/ceapi/...``): histórico datado de valor de
  mercado e histórico de transferências. O segundo só responde com cabeçalhos de XHR.

Tudo é cacheado em disco por chave natural (clube-temporada ou jogador), o que torna a coleta
retomável e reprodutível.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from lxml import html as lxml_html
from tqdm.auto import tqdm

from .http import BlockedError, FetchError, Throttle, get
from .io import read_json, read_text_gz, write_json, write_text_gz
from .logging_utils import get_logger

log = get_logger("transfermarkt")

BASE = "https://www.transfermarkt.com"
XHR_HEADERS = {"X-Requested-With": "XMLHttpRequest", "Accept": "application/json", "Referer": BASE + "/"}

_CLUB_HREF = re.compile(r"^/([a-z0-9-]+)/(?:kader|startseite)/verein/(\d+)/saison_id/(\d+)$")
_PLAYER_HREF = re.compile(r"^/([^/]+)/profil/spieler/(\d+)")
_VEREIN_ID = re.compile(r"/verein/(\d+)")
_DATE = re.compile(r"(\d{2})/(\d{2})/(\d{4})")
_DOB_WITH_AGE = re.compile(r"(\d{2}/\d{2}/\d{4})\s*\(\d+\)")
_HEIGHT = re.compile(r"^(\d)[,.](\d{2})\s*m$")
_VALUE = re.compile(r"€\s*([\d.,]+)\s*(k|m|bn)?", re.IGNORECASE)
_TAGS = re.compile(r"<[^>]+>")


# ----------------------------------------------------------------------------------------------
# parsers de valores escalares
# ----------------------------------------------------------------------------------------------
def parse_date(text: str | None) -> str | None:
    """Primeira data ``dd/mm/aaaa`` do texto, em ISO; None se ausente ou inválida."""
    if not text:
        return None
    match = _DATE.search(text)
    if not match:
        return None
    day, month, year = (int(g) for g in match.groups())
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def parse_value_eur(text: str | None) -> float | None:
    """``"€35.00m"`` → 35e6; ``"€500k"`` → 5e5; ``"-"`` → None."""
    if not text:
        return None
    match = _VALUE.search(text)
    if not match:
        return None
    number, suffix = match.groups()
    multiplier = {"k": 1e3, "m": 1e6, "bn": 1e9, None: 1.0}[suffix.lower() if suffix else None]
    try:
        return float(number.replace(",", "")) * multiplier
    except ValueError:
        return None


def parse_height_cm(text: str | None) -> float | None:
    """``"1,86m"`` → 186.0."""
    if not text:
        return None
    match = _HEIGHT.match(text.replace("\xa0", " ").strip())
    return float(f"{match.group(1)}{match.group(2)}") if match else None


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


# ----------------------------------------------------------------------------------------------
# páginas HTML
# ----------------------------------------------------------------------------------------------
def league_url(slug: str, competition_id: str, saison_id: str) -> str:
    return f"{BASE}/{slug}/startseite/wettbewerb/{competition_id}/saison_id/{saison_id}"


def squad_url(club_slug: str, club_id: int, saison_id: str) -> str:
    return f"{BASE}/{club_slug}/kader/verein/{club_id}/saison_id/{saison_id}/plus/1"


def parse_league_clubs(page_html: str, saison_id: str) -> list[dict[str, Any]]:
    """Clubes participantes de uma liga-temporada (id, slug e nome oficial no Transfermarkt)."""
    tree = lxml_html.fromstring(page_html)
    clubs: dict[int, dict[str, Any]] = {}
    table = tree.xpath("//table[contains(@class, 'items')]")
    scope = table[0] if table else tree
    for anchor in scope.xpath(".//a[@href]"):
        match = _CLUB_HREF.match(anchor.get("href"))
        if not match or match.group(3) != saison_id:
            continue
        slug, club_id = match.group(1), int(match.group(2))
        name = anchor.get("title") or _clean(anchor.text_content())
        if club_id not in clubs and name:
            clubs[club_id] = {"tm_club_id": club_id, "tm_club_slug": slug, "tm_team": name}
    return list(clubs.values())


def parse_squad(page_html: str) -> list[dict[str, Any]]:
    """Linhas do elenco detalhado (``/plus/1``).

    As células são identificadas pelo **conteúdo** (data com idade, bandeiras, altura, pé...) em
    vez da posição, porque a ordem das colunas muda entre a temporada corrente (com "Contract") e
    as passadas (com "Current club"). A célula do jogador contém uma tabela aninhada, por isso só
    as células filhas diretas da linha são lidas.
    """
    tree = lxml_html.fromstring(page_html)
    tables = tree.xpath("//table[contains(concat(' ', normalize-space(@class), ' '), ' items ')]")
    if not tables:
        return []
    rows: list[dict[str, Any]] = []
    for tr in tables[0].xpath("./tbody/tr[contains(@class, 'odd') or contains(@class, 'even')]"):
        anchor = next((a for a in tr.xpath(".//td[contains(@class,'hauptlink')]//a[@href]")
                       if _PLAYER_HREF.match(a.get("href"))), None)
        if anchor is None:
            continue
        slug, player_id = _PLAYER_HREF.match(anchor.get("href")).groups()
        inline_rows = tr.xpath(".//table[contains(@class,'inline-table')]//tr")
        record: dict[str, Any] = {
            "tm_player_id": int(player_id),
            "tm_player_slug": slug,
            "tm_player": _clean(anchor.text_content()),
            "tm_position": _clean(inline_rows[1].text_content()) if len(inline_rows) > 1 else None,
            "tm_birth_date": None, "tm_nationalities": None, "tm_height_cm": None, "tm_foot": None,
            "tm_joined_date_raw": None, "tm_signed_from": None, "tm_squad_value_eur": None,
        }
        seen_dob = False
        for td in tr.xpath("./td"):
            text = _clean(td.text_content())
            classes = td.get("class") or ""
            if "posrela" in classes or "rueckennummer" in classes:
                continue
            if not seen_dob and _DOB_WITH_AGE.search(text):
                record["tm_birth_date"] = parse_date(text)
                seen_dob = True
            elif td.xpath(".//img[contains(@class,'flaggenrahmen')]") and record["tm_nationalities"] is None:
                record["tm_nationalities"] = "|".join(img.get("title") for img in
                                                      td.xpath(".//img[contains(@class,'flaggenrahmen')]"))
            elif _HEIGHT.match(text):
                record["tm_height_cm"] = parse_height_cm(text)
            elif text in ("left", "right", "both"):
                record["tm_foot"] = text
            elif _DATE.fullmatch(text) and seen_dob and record["tm_joined_date_raw"] is None:
                record["tm_joined_date_raw"] = parse_date(text)
            elif "rechts" in classes and "€" in text:
                record["tm_squad_value_eur"] = parse_value_eur(text)
            else:
                link = td.xpath(".//a[@title]")
                if link and ":" in (link[0].get("title") or "") and record["tm_signed_from"] is None:
                    record["tm_signed_from"] = link[0].get("title").split(":")[0].strip()
        rows.append(record)
    return rows


_PROFILE_FIELDS = {
    "Date of birth/Age": "birth_date",
    "Height": "height",
    "Citizenship": "citizenship",
    "Position": "position",
    "Foot": "foot",
    "Current club": "current_club",
    "Joined": "joined_current_club",
    "Contract expires": "contract_expires",
    "Last contract extension": "last_contract_extension",
    "On loan from": "on_loan_from",
    "Contract there expires": "parent_contract_expires",
    "Contract option": "contract_option",
}
_PROFILE_DATES = {"birth_date", "joined_current_club", "contract_expires", "last_contract_extension",
                  "parent_contract_expires"}


def parse_profile(page_html: str) -> dict[str, Any]:
    """Pares rótulo→valor da tabela de dados do perfil, normalizados.

    Apenas o contrato **atual** é publicado; a validade dele para temporadas passadas é decidida
    no notebook 03 a partir de `joined_current_club`, `last_contract_extension` e das transferências.
    """
    tree = lxml_html.fromstring(page_html)
    raw: dict[str, str] = {}
    for label in tree.xpath("//span[contains(@class,'info-table__content--regular')]"):
        value = label.getnext()
        if value is None or "info-table__content--bold" not in (value.get("class") or ""):
            continue
        key = _clean(label.text_content()).rstrip(":")
        text = " | ".join(t for t in (_clean(x) for x in value.itertext()) if t) if key == "Citizenship" \
            else _clean(value.text_content())
        raw[key] = text

    profile: dict[str, Any] = {name: None for name in _PROFILE_FIELDS.values()}
    for label, name in _PROFILE_FIELDS.items():
        if label in raw:
            profile[name] = parse_date(raw[label]) if name in _PROFILE_DATES else (raw[label] or None)
    profile["height_cm"] = parse_height_cm(profile.pop("height"))
    profile["raw_fields"] = raw
    return profile


# ----------------------------------------------------------------------------------------------
# endpoints JSON
# ----------------------------------------------------------------------------------------------
def parse_market_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Histórico datado de valor de mercado, com o clube do jogador em cada avaliação."""
    out = []
    for entry in payload.get("list", []):
        when = parse_date(entry.get("datum_mw"))
        if when is None or entry.get("y") is None:
            continue
        out.append({"mv_date": when, "mv_value_eur": float(entry["y"]),
                    "mv_club": entry.get("verein") or None, "mv_age": entry.get("age")})
    return out


def parse_transfers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in payload.get("transfers", []):
        fee_text = _clean(_TAGS.sub(" ", item.get("fee") or ""))
        lowered = fee_text.lower()
        src, dst = item.get("from") or {}, item.get("to") or {}
        src_id, dst_id = _VEREIN_ID.search(src.get("href") or ""), _VEREIN_ID.search(dst.get("href") or "")
        out.append({
            "transfer_date": item.get("dateUnformatted") or parse_date(item.get("date")),
            "transfer_season": item.get("season"),
            "from_club": src.get("clubName"),
            "from_club_id": int(src_id.group(1)) if src_id else None,
            "to_club": dst.get("clubName"),
            "to_club_id": int(dst_id.group(1)) if dst_id else None,
            "fee_text": fee_text or None,
            "fee_eur": parse_value_eur(fee_text) if "€" in fee_text else None,
            "is_loan": "loan" in lowered and "end of loan" not in lowered,
            "is_end_of_loan": "end of loan" in lowered,
            "market_value_text": item.get("marketValue"),
            "upcoming": bool(item.get("upcoming")),
        })
    return out


# ----------------------------------------------------------------------------------------------
# cliente com cache
# ----------------------------------------------------------------------------------------------
class TransfermarktClient:
    def __init__(self, session: requests.Session, throttle: Throttle, cache_dir: Path) -> None:
        self.session = session
        self.throttle = throttle
        self.cache_dir = cache_dir
        self.network_fetches = 0

    def _html(self, url: str, cache_path: Path) -> str:
        cached = read_text_gz(cache_path)
        if cached is not None:
            return cached
        text = get(self.session, url, self.throttle).text
        write_text_gz(cache_path, text)
        self.network_fetches += 1
        return text

    def _json(self, url: str, cache_path: Path, headers: dict[str, str] | None = None) -> Any:
        cached = read_json(cache_path)
        if cached is not None:
            return cached
        payload = get(self.session, url, self.throttle, headers=headers).json()
        write_json(cache_path, payload)
        self.network_fetches += 1
        return payload

    def league_clubs(self, slug: str, competition_id: str, saison_id: str) -> list[dict[str, Any]]:
        path = self.cache_dir / "leagues" / f"{competition_id}_{saison_id}.html.gz"
        clubs = parse_league_clubs(self._html(league_url(slug, competition_id, saison_id), path), saison_id)
        if not clubs:
            path.unlink(missing_ok=True)
            raise FetchError(f"nenhum clube em {competition_id} {saison_id}")
        return clubs

    def squad(self, club_slug: str, club_id: int, saison_id: str) -> list[dict[str, Any]]:
        path = self.cache_dir / "squads" / f"{club_id}_{saison_id}.html.gz"
        rows = parse_squad(self._html(squad_url(club_slug, club_id, saison_id), path))
        if not rows:
            path.unlink(missing_ok=True)
            raise FetchError(f"elenco vazio: clube {club_id} temporada {saison_id}")
        return rows

    def market_values(self, player_id: int) -> list[dict[str, Any]]:
        path = self.cache_dir / "market_values" / f"{player_id}.json"
        return parse_market_values(self._json(f"{BASE}/ceapi/marketValueDevelopment/graph/{player_id}", path))

    def profile(self, player_id: int, slug: str) -> dict[str, Any]:
        """Perfil parseado (o HTML de ~120 kB por jogador não é guardado; só os campos)."""
        path = self.cache_dir / "profiles" / f"{player_id}.json"
        cached = read_json(path)
        if cached is not None:
            return cached
        text = get(self.session, f"{BASE}/{slug}/profil/spieler/{player_id}", self.throttle).text
        profile = {"tm_player_id": player_id, **parse_profile(text),
                   "scraped_at": datetime.now().date().isoformat()}
        write_json(path, profile)
        self.network_fetches += 1
        return profile

    def transfers(self, player_id: int) -> list[dict[str, Any]]:
        path = self.cache_dir / "transfers" / f"{player_id}.json"
        payload = self._json(f"{BASE}/ceapi/transferHistory/list/{player_id}", path, headers=XHR_HEADERS)
        return parse_transfers(payload)

    def is_cached(self, kind: str, player_id: int) -> bool:
        return (self.cache_dir / kind / f"{player_id}.json").exists()


def fetch_for_players(
    client: TransfermarktClient,
    players: pd.DataFrame,
    kind: str,
    fetch_one: Callable[[int, str], Any],
    failed_path: Path,
    max_fetches: int | None = None,
) -> tuple[dict[int, Any], set[int]]:
    """Executa `fetch_one` para cada jogador, de forma retomável.

    Cache é lido sem limite; requisições de rede param em `max_fetches`. Falhas pontuais são
    registradas em `failed_path` e re-tentadas na próxima execução. `BlockedError` encerra as
    requisições de rede da execução (insistir pioraria o bloqueio), mas o cache dos demais jogadores
    continua sendo lido.
    """
    previous_failures = set((read_json(failed_path) or {}).get(kind, []))
    pending = [int(pid) for pid in players["tm_player_id"] if not client.is_cached(kind, int(pid))]
    log.info("coleta por jogador", extra={"kind": kind, "players": len(players),
                                          "to_fetch": len(pending), "retrying": len(previous_failures)})
    results: dict[int, Any] = {}
    failures: set[int] = set()
    start_fetches = client.network_fetches
    blocked = False
    for row in tqdm(players.itertuples(index=False), total=len(players), desc=kind):
        pid = int(row.tm_player_id)
        if not client.is_cached(kind, pid) and (blocked or (
                max_fetches is not None and client.network_fetches - start_fetches >= max_fetches)):
            continue
        try:
            results[pid] = fetch_one(pid, row.tm_player_slug)
        except BlockedError:
            log.error("bloqueio detectado; rede suspensa nesta execução, cache segue sendo lido",
                      extra={"kind": kind})
            failures.add(pid)
            blocked = True
        except FetchError as exc:
            log.warning("falha ao buscar jogador", extra={"kind": kind, "tm_player_id": pid, "error": str(exc)})
            failures.add(pid)
    record = read_json(failed_path) or {}
    record[kind] = sorted(failures)
    write_json(failed_path, record)
    missing = len(players) - len(results) - len(failures)
    log.info("coleta concluída", extra={"kind": kind, "resolved": len(results), "failed": len(failures),
                                         "not_attempted": missing})
    if missing:
        log.warning("coleta INCOMPLETA — reexecute a célula para continuar", extra={"kind": kind, "missing": missing})
    return results, failures


def collect_squads(
    client: TransfermarktClient,
    leagues: Iterable[tuple[str, str, str]],
    seasons: Iterable[tuple[str, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Elencos de todas as ligas-temporadas.

    `leagues` = (chave da liga, slug TM, id da competição); `seasons` = (código, saison_id).
    Retorna o DataFrame e a lista de clubes-temporada que falharam (para reexecução).
    """
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    seasons = list(seasons)
    for league_key, slug, competition_id in leagues:
        for season_code, saison_id in seasons:
            try:
                clubs = client.league_clubs(slug, competition_id, saison_id)
            except BlockedError:
                raise
            except FetchError as exc:
                failures.append({"league": league_key, "season": season_code, "tm_team": "<liga>",
                                 "error": str(exc)})
                log.warning("página da liga falhou", extra={"league": league_key, "season": season_code,
                                                            "error": str(exc)})
                continue
            for club in tqdm(clubs, desc=f"{league_key} {season_code}", leave=False):
                try:
                    rows = client.squad(club["tm_club_slug"], club["tm_club_id"], saison_id)
                except BlockedError:
                    raise
                except FetchError as exc:
                    failures.append({"league": league_key, "season": season_code, **club, "error": str(exc)})
                    log.warning("elenco falhou", extra={"club": club["tm_team"], "season": season_code,
                                                        "error": str(exc)})
                    continue
                frame = pd.DataFrame(rows)
                frame["league"], frame["season"] = league_key, season_code
                for key, value in club.items():
                    frame[key] = value
                frames.append(frame)
            log.info("elencos TM", extra={"league": league_key, "season": season_code, "clubs": len(clubs)})
    squads = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not squads.empty:
        squads = squads.drop_duplicates(["league", "season", "tm_club_id", "tm_player_id"]).reset_index(drop=True)
    return squads, failures
