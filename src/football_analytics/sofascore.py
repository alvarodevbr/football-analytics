"""Sofascore: vetor tático por temporada (passes, dribles, duelos, ações defensivas).

Repõe as métricas que o FBref deixou de publicar em 2026. A API de estatísticas por
torneio-temporada é consultada **por clube** (``filters=team.in.<id>``), o que gera listas com
menos de 100 linhas (sem paginação, logo determinísticas) no mesmo grão jogador-clube do FBref.
Cada resposta bruta fica em cache JSON; reexecuções não tocam a rede.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from .browser import BrowserJsonClient
from .config import ExtractionConfig, League, Season
from .http import Throttle
from .io import read_json, write_json
from .logging_utils import get_logger

log = get_logger("sofascore")

HOME = "https://www.sofascore.com/"
API = "https://www.sofascore.com/api/v1"


def _seasons_url(tournament_id: int) -> str:
    return f"{API}/unique-tournament/{tournament_id}/seasons"


def _teams_url(tournament_id: int, season_id: int) -> str:
    return f"{API}/unique-tournament/{tournament_id}/season/{season_id}/teams"


def _stats_url(tournament_id: int, season_id: int, team_id: int, fields: Iterable[str]) -> str:
    return (f"{API}/unique-tournament/{tournament_id}/season/{season_id}/statistics"
            f"?limit=100&offset=0&order=-minutesPlayed&accumulation=total"
            f"&fields={'%2C'.join(fields)}&filters=team.in.{team_id}")


class SofascoreCollector:
    """Coleta com cache; o navegador só é aberto quando há algo a buscar."""

    def __init__(self, cfg: ExtractionConfig) -> None:
        self.cfg = cfg
        self.cache = cfg.paths.sofascore
        self._client: BrowserJsonClient | None = None
        self.network_fetches = 0

    def _get(self, url: str, cache_path: Path) -> Any:
        cached = read_json(cache_path)
        if cached is not None:
            return cached
        if self._client is None:
            throttle = Throttle(self.cfg.sofascore_delay_seconds, self.cfg.request_jitter_seconds)
            self._client = BrowserJsonClient(HOME, throttle, headless=self.cfg.browser_headless).__enter__()
        payload = self._client.get_json(url)
        write_json(cache_path, payload)
        self.network_fetches += 1
        return payload

    def close(self) -> None:
        if self._client is not None:
            self._client.__exit__(None, None, None)
            self._client = None

    def season_id(self, league: League, season: Season) -> int:
        payload = self._get(_seasons_url(league.sofascore_tournament_id),
                            self.cache / "seasons" / f"{league.sofascore_tournament_id}.json")
        for item in payload["seasons"]:
            if item["year"] == season.sofascore_year:
                return int(item["id"])
        raise KeyError(f"Sofascore: temporada {season.sofascore_year} ausente para {league.key}")

    def teams(self, league: League, season_id: int) -> list[dict[str, Any]]:
        payload = self._get(_teams_url(league.sofascore_tournament_id, season_id),
                            self.cache / "teams" / f"{league.sofascore_tournament_id}_{season_id}.json")
        return [{"id": int(t["id"]), "name": t["name"]} for t in payload["teams"]]

    def team_stats(self, league: League, season_id: int, team_id: int) -> list[dict[str, Any]]:
        path = self.cache / "stats" / f"{league.sofascore_tournament_id}_{season_id}_{team_id}.json"
        payload = self._get(_stats_url(league.sofascore_tournament_id, season_id, team_id,
                                       self.cfg.sofascore_fields), path)
        if payload.get("pages", 1) > 1:
            log.warning("lista de clube paginada (inesperado)", extra={"team_id": team_id, "pages": payload["pages"]})
        return payload["results"]

    def collect(self, leagues: Iterable[League], seasons: Iterable[Season]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        try:
            for league in leagues:
                for season in seasons:
                    sid = self.season_id(league, season)
                    teams = self.teams(league, sid)
                    for team in teams:
                        if (self.cfg.max_fetches_per_run is not None
                                and self.network_fetches >= self.cfg.max_fetches_per_run):
                            log.warning("limite de requisições atingido; reexecute para continuar")
                            return _to_frame(rows, self.cfg.sofascore_fields)
                        for item in self.team_stats(league, sid, team["id"]):
                            rows.append({
                                "league": league.key,
                                "season": season.code,
                                "sofascore_season_id": sid,
                                "sofascore_player_id": int(item["player"]["id"]),
                                "ss_player": item["player"]["name"],
                                "sofascore_team_id": team["id"],
                                "ss_team": team["name"],
                                **{f: item.get(f) for f in self.cfg.sofascore_fields},
                            })
                    log.info("Sofascore liga-temporada", extra={"league": league.key, "season": season.code,
                                                                 "teams": len(teams)})
        finally:
            self.close()
        return _to_frame(rows, self.cfg.sofascore_fields)


def _to_frame(rows: list[dict[str, Any]], fields: Iterable[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rename = {f: f"ss_{_snake(f)}" for f in fields}
    df = df.rename(columns=rename)
    numeric = list(rename.values())
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def _snake(name: str) -> str:
    out = []
    for ch in name:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out)
