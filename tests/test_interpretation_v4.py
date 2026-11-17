import numpy as np
import pandas as pd
import pytest

from football_analytics import interpretation as it
from football_analytics import modeling as md


def _data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "season": rng.choice(["1819", "1920", "2021"], n), "x": rng.normal(size=n), "age": rng.uniform(18, 34, n),
        "xg_xa_p90": rng.gamma(2, 0.1, n), "value_gap_T": rng.normal(size=n),
        "position_group": rng.choice(["DEF", "MEI", "ATA"], n), "age_band": rng.choice(["16–23", "24–28"], n),
        "market_log_return_lag1_T": 0.0,
    })
    d["market_log_return_lag1_T"] = d["season"].map({"1819": 0.05, "1920": -0.2, "2021": 0.1})
    d["y"] = 0.4 * d["x"] - 0.3 * d["value_gap_T"] + d["season"].map({"1819": 0.0, "1920": -0.4, "2021": 0.1}) \
        + rng.normal(0, 0.2, n)
    return d


def test_ensemble_shap_is_additive_with_drift_wrapper_and_interaction_ridge():
    d = _data()
    num = ["x", "age", "xg_xa_p90", "value_gap_T", "market_log_return_lag1_T"]
    ridge_ic = md.DriftAdjusted(md._ridge_ic({"alpha": 1.0}, num, ["position_group"]), mode="regime")
    xgb = md.DriftAdjusted(md.Pipeline(md.Preprocessor(num, ["position_group"]),
                                       __import__("xgboost").XGBRegressor(n_estimators=30, max_depth=3)), mode="regime")
    comps = {"ridge_ic": ridge_ic.fit(d, d["y"].to_numpy()), "xgb": xgb.fit(d, d["y"].to_numpy())}
    final = it.fold_model(1, comps)
    phi, base = it.ensemble_shap(final, d)  # a verificação de aditividade está dentro
    assert not any("×" in c for c in phi.columns)            # interações somadas à variável original
    assert phi["market_log_return_lag1_T"].abs().sum() > 0   # o drift de regime é atribuído ao regime
    np.testing.assert_allclose(base + phi.sum(axis=1), final.predict(d), atol=1e-3)


def test_coalition_shap_sums_members():
    phi = pd.DataFrame({"age": [1.0, -2.0], "age_sq": [0.5, 0.5], "age_band": [0.1, 0.0], "x": [3.0, 1.0]})
    out = it.coalition_shap(phi, {"Ciclo etário": ["age", "age_sq", "age_band"]})
    assert list(out.columns) == ["x", "Ciclo etário"]
    np.testing.assert_allclose(out.sum(axis=1), phi.sum(axis=1))


def test_outcome_value_reason_zeroes_only_career_end():
    sel = pd.DataFrame({"value_T": [10.0, 10.0, 10.0], "dv_true": [5.0, np.nan, np.nan],
                        "value_last_obs": [np.nan, 4.0, 6.0],
                        "target_missing_reason": ["observed", "career_end", "stale"]})
    assert it.outcome_value(sel, "reason").tolist() == [15.0, 0.0, 6.0]
    assert it.outcome_value(sel, "zero").tolist() == [15.0, 0.0, 0.0]


def test_break_even_return_with_holding_costs():
    assert it.break_even_return(0.1, 0.1) == pytest.approx(0.2222, abs=1e-4)
    h1 = it.break_even_return(0.1, 0.1, horizon=1, rate=0.05, holding=0.10)
    h2 = it.break_even_return(0.1, 0.1, horizon=2, rate=0.05, holding=0.10)
    assert h1 == pytest.approx((1.1 * 1.05 + 0.1) / 0.9 - 1)
    assert h2 > h1 > it.break_even_return(0.1, 0.1)


def _cands(n: int = 40, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"value_T": rng.uniform(1e6, 15e6, n), "score": rng.normal(30, 20, n),
                         "q10": rng.normal(-20, 15, n), "position_group": rng.choice(["DEF", "MEI", "ATA"], n),
                         "age_band": rng.choice(["16–23", "24–28", "29–32"], n)})


def test_budget_milp_respects_constraints_and_beats_greedy_objective():
    c = _cands()
    caps = {"position_group": 2, "age_band": 3}
    sel = it.budget_milp(c, "score", 40e6, buy_cost=0.1, max_price=15e6, max_players=6, max_per_group=caps)
    assert (sel["value_T"] * 1.1).sum() <= 40e6 + 1e-6
    assert len(sel) <= 6 and sel["position_group"].value_counts().max() <= 2 and sel["age_band"].value_counts().max() <= 3
    greedy = it.budget_greedy(c.assign(value_T=c["value_T"] * 1.1), "score", 40e6, max_players=6, max_per_group=caps)
    assert sel["score"].clip(lower=0).sum() >= greedy["score"].clip(lower=0).sum() - 1e-9
    risky = it.budget_milp(c, "score", 40e6, buy_cost=0.1, max_players=6, max_per_group=caps,
                           downside="q10", hurdle=0.222, lam=1.0)
    assert risky["q10"].mean() >= sel["q10"].mean() - 1e-9  # a penalidade puxa a carteira para P10 mais alto


def test_buyable_excludes_elite_clubs_unless_low_minutes():
    c = pd.DataFrame({"season": ["2324"] * 4, "fbref_team_id": ["a", "a", "b", "c"],
                      "club_value_rank_pct": [0.99, 0.99, 0.5, 0.2], "minutes_share_pct": [80.0, 10.0, 80.0, 80.0]})
    assert it.buyable(c, top_clubs=1).tolist() == [False, True, True, True]


def test_acquisition_premium_uses_only_past_transfers():
    rows = pd.DataFrame({"tm_player_id": [1, 2, 3], "season": ["2122", "2122", "2223"], "value_T": [10e6, 5e6, 8e6],
                         "age_band": ["16–23"] * 3, "elite_seller": [False] * 3})
    transfers = pd.DataFrame({"tm_player_id": [1, 2, 3], "transfer_date": ["2022-08-01", "2022-07-15", "2023-08-01"],
                              "fee_eur": [15e6, 5e6, 20e6], "is_loan": [False, False, False]})
    early = it.acquisition_premium(rows, transfers, pd.Timestamp("2022-12-31"), min_n=1)
    assert early["n_total"].iloc[0] == 2 and early["premio"].iloc[0] == pytest.approx(1.25)
    late = it.acquisition_premium(rows, transfers, pd.Timestamp("2023-12-31"), min_n=1)
    assert late["n_total"].iloc[0] == 3


def test_wilson_interval():
    lo, hi = it.wilson_interval(67, 100)
    assert lo < 0.67 < hi and hi - lo < 0.2
