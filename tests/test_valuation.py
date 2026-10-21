import numpy as np
import pandas as pd

from football_analytics.valuation import attach_values


def _mv(rows):
    return pd.DataFrame(rows, columns=["tm_player_id", "mv_date", "mv_value_eur", "mv_club"])


def test_value_is_last_before_anchor_never_after():
    mv = _mv([
        (1, "2023-12-20", 10e6, "A"),
        (1, "2024-06-05", 15e6, "A"),
        (1, "2024-07-10", 40e6, "B"),   # depois da âncora de 2023/24: não pode vazar para T
        (1, "2025-06-01", 30e6, "B"),
        (1, "2026-06-10", 25e6, "B"),
    ])
    rows = pd.DataFrame({"tm_player_id": [1], "season": ["2324"]})
    out = attach_values(rows, mv)
    assert out.loc[0, "tm_value_T"] == 15e6
    assert out.loc[0, "tm_value_date_T"] == pd.Timestamp("2024-06-05")
    assert out.loc[0, "tm_value_days_stale_T"] == 25
    assert out.loc[0, "tm_value_T1"] == 30e6
    assert out.loc[0, "tm_value_club_T1"] == "B"
    assert out.loc[0, "tm_value_T2"] == 25e6


def test_stale_value_becomes_nan_and_unmatched_rows_survive():
    mv = _mv([(2, "2019-01-15", 5e6, "C")])
    rows = pd.DataFrame({"tm_player_id": [2, np.nan, 2], "season": ["1819", "1920", "2021"]})
    out = attach_values(rows, mv, max_staleness_days=183)
    assert out.loc[0, "tm_value_T"] == 5e6           # 166 dias antes de 30/06/2019
    assert np.isnan(out.loc[1, "tm_value_T"])         # sem id
    assert np.isnan(out.loc[2, "tm_value_T"])         # última avaliação velha demais
    assert len(out) == 3 and list(out["season"]) == ["1819", "1920", "2021"]
