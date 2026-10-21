"""Seleção do valor de mercado vigente em uma data-âncora, sem olhar para o futuro."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import Season

HORIZON_SUFFIX = {-2: "Tm2", -1: "Tm1", 0: "T", 1: "T1", 2: "T2"}


def anchor_date(season_code: str, horizon: int) -> pd.Timestamp:
    """30/jun do fim da temporada T+k (fim da janela de avaliação da temporada)."""
    return pd.Timestamp(Season(season_code).offset(horizon).end_anchor)


def attach_values(
    rows: pd.DataFrame,
    market_values: pd.DataFrame,
    horizons: Iterable[int] = (0, 1, 2),
    max_staleness_days: int = 183,
) -> pd.DataFrame:
    """Acrescenta, para cada horizonte k, o **último** valor publicado até a âncora de T+k.

    Usa-se a avaliação mais recente com data ≤ âncora (``merge_asof`` para trás), limitada a
    `max_staleness_days`: o valor de T nunca incorpora informação posterior ao fim da temporada,
    e uma avaliação velha demais (jogador sem atualização, aposentado) vira NaN em vez de ser
    tratada como vigente.

    `rows` precisa de ``tm_player_id`` e ``season``; `market_values`, de ``tm_player_id``,
    ``mv_date``, ``mv_value_eur`` e ``mv_club``. Linhas sem ``tm_player_id`` recebem NaN.
    """
    mv = market_values[["tm_player_id", "mv_date", "mv_value_eur", "mv_club"]].copy()
    mv["tm_player_id"] = mv["tm_player_id"].astype("int64")
    mv["mv_date"] = pd.to_datetime(mv["mv_date"])
    mv = mv.dropna(subset=["mv_date", "mv_value_eur"]).sort_values("mv_date")

    out = rows.copy()
    out["_row"] = range(len(out))
    known = out[out["tm_player_id"].notna()][["_row", "tm_player_id", "season"]].copy()
    known["tm_player_id"] = known["tm_player_id"].astype("int64")

    for k in horizons:
        suffix = HORIZON_SUFFIX[k]
        probe = known.assign(anchor=[anchor_date(s, k) for s in known["season"]]).sort_values("anchor")
        matched = pd.merge_asof(
            probe, mv, left_on="anchor", right_on="mv_date", by="tm_player_id",
            direction="backward", tolerance=pd.Timedelta(days=max_staleness_days),
        )
        matched[f"tm_value_days_stale_{suffix}"] = (matched["anchor"] - matched["mv_date"]).dt.days
        matched = matched.rename(columns={
            "mv_value_eur": f"tm_value_{suffix}",
            "mv_date": f"tm_value_date_{suffix}",
            "mv_club": f"tm_value_club_{suffix}",
        })[["_row", f"tm_value_{suffix}", f"tm_value_date_{suffix}", f"tm_value_club_{suffix}",
            f"tm_value_days_stale_{suffix}"]]
        out = out.merge(matched, on="_row", how="left")
    return out.sort_values("_row").drop(columns="_row").reset_index(drop=True)
