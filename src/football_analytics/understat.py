"""Understat: família de métricas de gols esperados (xG, npxG, xA, xGChain, xGBuildup).

Supre a ausência de xG/xA no FBref desde 2026. O soccerdata acessa o Understat por HTTP simples,
sem navegador, e devolve uma linha por (liga, temporada, clube, jogador).
"""

from __future__ import annotations

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
