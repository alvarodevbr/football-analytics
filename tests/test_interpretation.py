import numpy as np
import pandas as pd

from football_analytics import interpretation as it
from football_analytics import modeling as md


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "position_group": "DEF",
                      "league": rng.choice(["L1", "L2"], n), "season": np.where(np.arange(n) < n // 2, "2223", "2324")})
    y = 0.8 * X["a"] - 0.3 * X["b"] + 0.2 * (X["league"] == "L2") + rng.normal(scale=0.1, size=n)
    return X, y.to_numpy()


def test_ensemble_shap_is_additive():
    X, y = _data()
    comps = {
        "xgb": md._xgb({"max_depth": 3}, ["a", "b"], ["league"]).fit(X, y),
        "ridge": md._ridge({"alpha": 1.0}, ["a", "b"], ["league"]).fit(X, y),
    }
    final = it.FinalModel(1, comps, {})
    phi, base = it.ensemble_shap(final, X)          # levanta AssertionError se não for aditivo
    assert list(phi.columns) == ["a", "b", "league=L2"]
    collapsed = it.collapse_dummies(phi, ["league"])
    assert "league" in collapsed and np.allclose(collapsed.sum(axis=1), phi.sum(axis=1))
    imp = it.importance(collapsed)["mean_abs_shap"]
    assert imp.idxmax() == "a"


def test_portfolio_functions():
    cands = pd.DataFrame({"value_T": [10e6, 5e6, 2e6, 1e6], "dv_true": [5e6, -1e6, 2e6, 0.0],
                          "dv_pct_true": [50.0, -20.0, 100.0, 0.0], "y_true": [0.4, -0.2, 0.7, 0.0],
                          "score": [0.9, 0.1, 0.8, 0.5]})
    m = it.portfolio_metrics(cands.iloc[[0, 2]])
    assert m["roi_carteira_pct"] == 7e6 / 12e6 * 100 and m["taxa_valorizacao"] == 1.0
    sel = it.budget_greedy(cands, "score", budget=11e6)
    assert list(sel.index) == [0, 3]                 # 10 mi + 1 mi; o de 2 mi não cabe depois do primeiro
    assert list(it.top_k(cands, "score", 2).index) == [0, 2]
    rnd = it.random_portfolios(cands, 2, 50, np.random.default_rng(0))
    assert len(rnd) == 50 and rnd["taxa_valorizacao"].between(0, 1).all()
    var, cvar = it.var_cvar(np.array([-50.0, -10, 0, 10, 20, 30, 40, 50, 60, 70]), 0.1)
    assert var <= -10 and cvar <= var


def test_backtest_costs_missing_outcomes_and_robustness():
    sel = pd.DataFrame({"value_T": [10e6, 5e6, 2e6], "dv_true": [5e6, np.nan, 2e6],
                        "dv_pct_true": [50.0, np.nan, 100.0], "y_true": [0.4, np.nan, 0.7],
                        "value_last_obs": [np.nan, 4e6, np.nan]})
    excl = it.portfolio_metrics(sel)                                   # condicionado à sobrevivência
    assert excl["n"] == 2 and excl["sem_alvo"] == 1 and np.isclose(excl["roi_carteira_pct"], 7e6 / 12e6 * 100)
    zero = it.portfolio_metrics(sel, missing="zero")                   # quem saiu vale zero
    assert np.isclose(zero["roi_carteira_pct"], (19e6 - 17e6) / 17e6 * 100)
    last = it.portfolio_metrics(sel, missing="last")                   # último valor publicado
    assert np.isclose(last["roi_carteira_pct"], (23e6 - 17e6) / 17e6 * 100)
    cost = it.portfolio_metrics(sel, buy_cost=0.10, sell_cost=0.10)
    assert np.isclose(cost["roi_carteira_pct"], (19e6 * 0.9 - 12e6 * 1.1) / (12e6 * 1.1) * 100)
    be = it.break_even_buy_cost(sel)
    assert np.isclose(it.portfolio_metrics(sel, buy_cost=be)["roi_carteira_pct"], 0, atol=1e-9)
    assert np.isclose(it.roi_without_top(sel), 2e6 / 2e6 * 100)        # sem o maior ganho (5 mi)
    assert np.isclose(it.sign_test(6, 6), 1 / 64) and it.sign_test(3, 6) > 0.5


def test_budget_greedy_diversification():
    cands = pd.DataFrame({"value_T": [3e6] * 5, "score": [5, 4, 3, 2, 1],
                          "position_group": ["ATA", "ATA", "ATA", "DEF", "MEI"], "age_band": ["16–23"] * 5})
    sel = it.budget_greedy(cands, "score", budget=40e6, max_players=4, max_per_group={"position_group": 2})
    assert list(sel.index) == [0, 1, 3, 4]                             # o 3º atacante é pulado


def test_shap_group_shares_stability_and_bootstrap():
    rng = np.random.default_rng(0)
    phi = pd.DataFrame({"idade": rng.normal(0, 2, 400), "gols": rng.normal(0, 1, 400), "passes": rng.normal(0, 1, 400)})
    group_of = {"idade": "Demográfico", "gols": "Ofensivo", "passes": "Criação"}
    shares = it.group_shares(phi, group_of)
    assert np.isclose(shares.sum(), 100) and shares.idxmax() == "Demográfico"
    imps = {f: phi.abs().mean() * (1 + 0.01 * f) for f in (1, 2, 3)}
    stab = it.importance_stability(imps, top=3)
    assert len(stab) == 3 and np.allclose(stab["tau_kendall"], 1.0)
    clusters = np.repeat(np.arange(40), 10)
    by = pd.Series(np.where(np.arange(400) < 200, "16–23", "33+"))
    ci = it.group_share_bootstrap(phi, group_of, by, clusters, n_boot=200)
    assert set(ci["nivel"]) == {"16–23", "33+"} and (ci["ic95_inf"] <= ci["participacao_pct"]).all()
    assert (ci["participacao_pct"] <= ci["ic95_sup"]).all()


def test_random_portfolios_apply_costs_and_missing():
    cands = pd.DataFrame({"value_T": [1e6, 1e6, 1e6], "dv_true": [1e6, 0.0, np.nan], "dv_pct_true": [100.0, 0.0, np.nan],
                          "value_last_obs": [np.nan, np.nan, 5e5]})
    rng = np.random.default_rng(0)
    full = it.random_portfolios(cands, 3, 5, rng, missing="zero")          # todos os três, o ausente vale zero
    assert np.allclose(full["roi_carteira_pct"], (3e6 - 3e6) / 3e6 * 100)
    excl = it.random_portfolios(cands, 3, 5, rng)                            # o ausente sai do sorteio
    assert np.allclose(excl["roi_carteira_pct"], (3e6 - 2e6) / 2e6 * 100)
    cost = it.random_portfolios(cands, 2, 5, rng, buy_cost=0.1, sell_cost=0.1)
    assert np.allclose(cost["roi_carteira_pct"], (3e6 * 0.9 - 2e6 * 1.1) / 2.2e6 * 100)
