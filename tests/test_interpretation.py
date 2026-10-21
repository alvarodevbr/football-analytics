import numpy as np
import pandas as pd

from football_analytics import interpretation as it
from football_analytics import modeling as md


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "position_group": "DEF",
                      "league": rng.choice(["L1", "L2"], n)})
    y = 0.8 * X["a"] - 0.3 * X["b"] + 0.2 * (X["league"] == "L2") + rng.normal(scale=0.1, size=n)
    return X, y.to_numpy()


def test_ensemble_shap_is_additive():
    X, y = _data()
    comps = {
        "xgb": md._xgb({"n_estimators": 50, "max_depth": 3}, ["a", "b"], ["league"]).fit(X, y),
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
