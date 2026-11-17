"""Understat: família de métricas de gols esperados (xG, npxG, xA, xGChain, xGBuildup).

Supre a ausência de xG/xA no FBref desde 2026. O soccerdata acessa o Understat por HTTP simples,
sem navegador, e devolve uma linha por (liga, temporada, clube, jogador).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import ExtractionConfig
from .logging_utils import get_logger

log = get_logger("understat")

_RENAME = {
    "player_id": "understat_player_id",
    "team_id": "understat_team_id",
    "player": "us_player",
    "team": "us_team",
    "position": "us_position",
}
_DROP = ["league_id", "season_id"]


def read_player_seasons(cfg: ExtractionConfig) -> pd.DataFrame:
    import soccerdata as sd

    reader = sd.Understat(
        leagues=list(cfg.league_keys),
        seasons=[s.code for s in cfg.seasons],
        data_dir=cfg.paths.soccerdata_dir / "Understat",
    )
    df = reader.read_player_season_stats().reset_index()
    df["season"] = df["season"].astype(str)
    df = df.drop(columns=[c for c in _DROP if c in df.columns]).rename(columns=_RENAME)
    metric_cols = [c for c in df.columns
                   if c not in {"league", "season", *_RENAME.values()}]
    df = df.rename(columns={c: f"us_{c}" for c in metric_cols})
    log.info("Understat lido", extra={"rows": len(df), "seasons": sorted(df["season"].unique().tolist())})
    return df


def _is_json(path: Path) -> bool:
    try:
        json.loads(path.read_bytes())
    except (ValueError, OSError):
        return False
    return True


_MATCH_RENAME = {
    "player_id": "understat_player_id",
    "team_id": "understat_team_id",
    "player": "us_player",
    "team": "us_team",
    "position": "us_position",
    "position_id": "us_position_order",
}
_MATCH_METRICS = ("minutes", "goals", "own_goals", "shots", "xg", "xg_chain", "xg_buildup", "assists", "xa",
                  "key_passes", "yellow_cards", "red_cards")


def read_player_matches(
    cfg: ExtractionConfig,
    chunk_size: int = 20,
    pause_seconds: float = 8.0,
    max_matches: int | None = None,
) -> pd.DataFrame:
    """Estatísticas por jogador e partida (escalações do ``getMatchData``), com a data do jogo.

    A coleta é retomável: só as partidas ausentes do cache vão à rede, em blocos de `chunk_size`
    separados por `pause_seconds` (o soccerdata não limita a taxa do Understat). Com `max_matches`,
    a execução para depois desse número de partidas novas e devolve apenas as já em cache.
    """
    import time

    import soccerdata as sd
    from tqdm.auto import tqdm

    data_dir = cfg.paths.soccerdata_dir / "Understat"
    reader = sd.Understat(
        leagues=list(cfg.league_keys),
        seasons=[s.code for s in cfg.seasons],
        data_dir=data_dir,
    )
    schedule = reader.read_schedule(include_matches_without_data=False).reset_index()
    game_ids = [int(g) for g in schedule["game_id"]]

    def cached(game_id: int) -> bool:
        return (data_dir / f"match_{game_id}.json").exists()

    pending = [g for g in game_ids if not cached(g)]
    if max_matches is not None:
        pending = pending[:max_matches]
    log.info("partidas do Understat", extra={"matches": len(game_ids), "to_fetch": len(pending)})
    for start in tqdm(range(0, len(pending), chunk_size), desc="Understat partidas"):
        chunk = pending[start:start + chunk_size]
        try:
            reader.read_player_match_stats(match_id=chunk)
        except json.JSONDecodeError:
            # Em manutenção o Understat responde 200 com uma página HTML, que o soccerdata gravaria
            # como cache da partida: apaga o que não for JSON e encerra a coleta desta execução.
            for game_id in chunk:
                path = data_dir / f"match_{game_id}.json"
                if path.exists() and not _is_json(path):
                    path.unlink()
            log.error("Understat respondeu sem JSON (manutenção?); coleta interrompida, reexecute depois")
            break
        time.sleep(pause_seconds)

    available = [g for g in game_ids if cached(g)]
    if len(available) < len(game_ids):
        log.warning("coleta de partidas INCOMPLETA — reexecute para continuar",
                    extra={"missing": len(game_ids) - len(available)})
    if not available:
        return pd.DataFrame(columns=["league", "season", "game_id", "date", "understat_team_id", "us_team",
                                     "understat_player_id", "us_player", "us_position", *[f"us_{m}" for m in _MATCH_METRICS]])
    df = reader.read_player_match_stats(match_id=available).reset_index()
    dates = schedule[["game_id", "date"]].assign(game_id=lambda d: d["game_id"].astype("int64"))
    df = (df.assign(game_id=df["game_id"].astype("int64"), season=df["season"].astype(str))
          .merge(dates, on="game_id", how="left", validate="many_to_one")
          .drop(columns=[c for c in ("league_id", "season_id", "game") if c in df.columns])
          .rename(columns=_MATCH_RENAME)
          .rename(columns={m: f"us_{m}" for m in _MATCH_METRICS}))
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    log.info("Understat por partida lido", extra={"rows": len(df), "matches": df["game_id"].nunique()})
    return df
