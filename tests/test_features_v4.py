import numpy as np
import pytest
import pandas as pd

from football_analytics import features as ft
from football_analytics import sample as sp


def test_tactical_role_maps_and_falls_back():
    pos = pd.Series(["Centre-Back", "Left-Back", "Defensive Midfield", "Right Winger", "Second Striker", None])
    grp = pd.Series(["DEF", "DEF", "MEI", "ATA", "ATA", "MEI"])
    assert sp.tactical_role(pos, grp).tolist() == ["CB", "FB", "DM", "AW", "CF", "CM"]


def test_tail_probability_separates_explosive_low_minutes_from_average_regular():
    rng = np.random.default_rng(0)
    n = 400
    exposure = pd.Series(rng.uniform(5, 38, n))
    counts = pd.Series(rng.poisson(rng.gamma(4, 0.075, n) * exposure)).astype(float)
    alpha, beta = ft._gamma_prior(counts, exposure, 1.0)
    # dois atletas com a MESMA média a posteriori: 2 jogos (jovem) e 25 jogos (regular)
    m = 1.1 * alpha / beta
    young, regular = m * (beta + 2) - alpha, m * (beta + 25) - alpha
    exposure = pd.concat([exposure, pd.Series([2.0, 25.0, 25.0])], ignore_index=True)
    counts = pd.concat([counts, pd.Series([young, regular, regular + 3])], ignore_index=True)
    groups = pd.Series("g", index=counts.index)
    shrunk = ft.shrink_counts(counts, exposure, groups)
    tail = ft.tail_probability_counts(counts, exposure, groups)
    assert tail.between(0, 1).all()
    assert abs(shrunk.iloc[-3] - shrunk.iloc[-2]) < 1e-3          # o encolhimento os iguala...
    assert abs(tail.iloc[-3] - tail.iloc[-2]) > 0.05               # ...a probabilidade de cauda, não
    assert tail.iloc[-1] > tail.iloc[-2]                           # mais gols com a mesma exposição


def test_tail_probability_means_bounds():
    rng = np.random.default_rng(1)
    w = pd.Series(rng.uniform(1, 38, 400))
    x = pd.Series(6.8 + rng.normal(0, 0.2, 400) + rng.normal(0, 0.6, 400) / np.sqrt(w))
    tail = ft.tail_probability_means(x, w, pd.Series("g", index=x.index))
    assert tail.between(0, 1).all()
    assert np.corrcoef(x, tail)[0, 1] > 0.5


def _injury_rows():
    return pd.DataFrame({"tm_player_id": [1, 1, 2, 3], "season": ["2223", "2324", "2324", "2324"]})


def test_injury_features_truncate_at_anchor_and_ignore_future():
    injuries = pd.DataFrame({
        "tm_player_id": [1, 1, 1, 2],
        "injury": ["Cruciate ligament tear", "Hamstring injury", "Knee injury", "Ill"],
        "injury_from": ["2024-04-01", "2023-01-10", "2024-08-10", "2022-03-01"],
        "injury_until": ["2025-01-31", "2023-01-20", "2024-09-01", None],
        "injury_days": [306, 11, 23, None],
    })
    out = ft.injury_features(_injury_rows(), injuries, collected={1, 2})
    t23, t24, p2, p3 = out.iloc[0], out.iloc[1], out.iloc[2], out.iloc[3]
    # 2022/23: só a lesão de jan/2023 (10 dias), nenhuma em curso
    assert t23["injury_days_T"] == 10 and t23["injured_at_anchor"] == 0
    # 2023/24: ruptura desde 01/04/2024 truncada em 30/06/2024 (90 dias); o retorno de 2025 não entra
    assert t24["injury_days_T"] == 90 and t24["injured_at_anchor"] == 1 and t24["long_injury_at_anchor"] == 1
    assert t24["injury_days_Tm1"] == 10 and t24["n_injuries_24m"] == 2
    # a lesão de ago/2024 é posterior à âncora de 2023/24
    assert t24["n_injuries_24m"] == 2
    # duração desconhecida em 2022 não fica "em curso" para sempre
    assert p2["injured_at_anchor"] == 0 and p2["injury_days_T"] == 0
    # não coletado: ausente, não zero
    assert p3.isna().all()


def test_performance_lags_trajectory_and_breakout():
    panel = pd.DataFrame({
        "fbref_player_id": ["a", "a", "a", "b", "b"],
        "season": ["2122", "2223", "2324", "2223", "2324"],
        "minutes": [300.0, 500.0, 2500.0, 2000.0, 2100.0],
        "minutes_share_pct": [10.0, 15.0, 70.0, 60.0, 62.0],
    })
    for c in ft.LAG_PERFORMANCE:
        if c not in panel:
            panel[c] = 0.0
    out = ft.performance_lags(panel)
    a24 = out.iloc[2]
    assert a24["lag2_minutes_share_pct"] == 10.0
    assert a24["minutes_share_accel"] == (70 - 15) - (15 - 10)
    assert a24["breakout_starter_T"] == 1 and out.iloc[4]["breakout_starter_T"] == 0
    assert out.iloc[0]["breakout_starter_T"] == 0  # sem T−1 nas Big Five não é "titularização"


def test_league_step_up():
    panel = pd.DataFrame({"season": ["2324"] * 4, "has_big5_lag1": [0.0, 0.0, 1.0, 1.0],
                          "minutes_share_pct": [80.0, 10.0, 90.0, 20.0]})
    assert ft.league_step_up(panel).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_target_missing_reason():
    rows = pd.DataFrame({"tm_player_id": [1, 2, 3, 4], "season": ["2223", "2223", "2223", "2425"],
                         "tm_value_T2": [5e6, np.nan, np.nan, np.nan]})
    mv = pd.DataFrame({"tm_player_id": [1, 2, 2, 3, 3],
                       "mv_date": ["2023-06-01", "2023-06-01", "2024-12-01", "2023-06-01", "2024-01-15"],
                       "mv_club": ["A", "A", "Retired", "A", "Some Club"], "mv_value_eur": [1, 1, 1, 1, 1]})
    profiles = pd.DataFrame({"tm_player_id": [1, 2, 3, 4], "current_club": ["A", "Retired", "Some Club", "B"]})
    reason = ft.target_missing_reason(rows, mv, profiles, horizon=2, cutoff=pd.Timestamp("2026-06-30"))
    assert reason.tolist() == ["observed", "career_end", "stale", "not_due"]


def test_valuation_gap_groups_by_club_and_adds_perf_only():
    rng = np.random.default_rng(3)
    n = 400
    panel = pd.DataFrame({
        "season": ["2324"] * n, "league": rng.choice(["ENG-Premier League", "FRA-Ligue 1"], n),
        "position_group": rng.choice(["DEF", "MEI", "ATA"], n), "role": rng.choice(["CB", "CM", "CF"], n),
        "fbref_team_id": rng.choice([f"c{i}" for i in range(20)], n),
        "age": rng.uniform(18, 34, n), "xg_xa_p90": rng.gamma(2, 0.1, n), "team_points_per_game": rng.uniform(0.8, 2.3, n),
    })
    panel["value_T"] = np.exp(15 + 2 * panel["xg_xa_p90"] + 0.5 * panel["team_points_per_game"] + rng.normal(0, 0.3, n))
    gap = ft.valuation_gap(panel, ["age", "xg_xa_p90", "team_points_per_game"], n_splits=5)
    assert set(gap.columns) == {"fair_log_value_T", "value_gap_T", "fair_log_value_perf_T", "value_gap_perf_T"}
    assert gap.notna().all().all()
    logv = np.log(panel["value_T"])
    np.testing.assert_allclose(gap["value_gap_T"], logv - gap["fair_log_value_T"])
    # sem o contexto do clube, o valor justo explica menos de ln(V_T)
    assert (gap["value_gap_perf_T"] ** 2).mean() > (gap["value_gap_T"] ** 2).mean()


def test_match_form_features_split_by_half_and_valuation_date():
    games = []
    gid = 0
    for d in pd.date_range("2023-08-15", "2024-05-20", freq="7D"):
        gid += 1
        for team in (1, 2):
            games.append({"season": "2324", "game_id": gid, "date": d, "understat_team_id": team})
    g = pd.DataFrame(games)
    rows_ = []
    for _, r in g.iterrows():
        # jogador 10: reserva no 1º turno, titular no 2º; jogador 20: chega em janeiro no time 2
        if r["understat_team_id"] == 1:
            rows_.append({**r, "understat_player_id": 10, "us_minutes": 90 if r["date"].month < 8 and r["date"].year == 2024 else 10,
                          "us_xg": 0.3, "us_xa": 0.1, "us_xg_chain": 0.5})
        elif r["date"].year == 2024:
            rows_.append({**r, "understat_player_id": 20, "us_minutes": 90, "us_xg": 0.1, "us_xa": 0.1, "us_xg_chain": 0.2})
    for pid in range(100, 130):  # população para a priori do encolhimento
        for _, r in g[g["understat_team_id"] == 2].iterrows():
            rows_.append({**r, "understat_player_id": pid, "us_minutes": 60, "us_xg": 0.05, "us_xa": 0.05, "us_xg_chain": 0.15})
    matches = pd.DataFrame(rows_)
    rows = pd.DataFrame({"season": "2324", "understat_player_id": [10, 20, *range(100, 130), 999],
                         "tm_value_date_T": ["2024-03-01", "2024-06-15", *["2024-06-15"] * 30, "2024-06-15"]})
    out = ft.match_form_features(rows, matches, pd.Series("g", index=rows.index))
    p10, p20, nobody = out.iloc[0], out.iloc[1], out.iloc[-1]
    assert p10["h2_minutes_share_T"] == pytest.approx(100.0) and p10["form_minutes_trend_T"] > 80
    assert p10["post_valuation_minutes_T"] > 0 and p20["post_valuation_minutes_T"] == 0  # avaliação após a última partida
    assert p20["debut_second_half_T"] == 1 and p10["debut_second_half_T"] == 0
    assert nobody.isna().all()
    assert out["h2_xg_chain_p90_T"].iloc[0] > out["h2_xg_chain_p90_T"].iloc[1]
