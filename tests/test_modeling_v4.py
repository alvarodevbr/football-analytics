import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from football_analytics import modeling as md


def _panel(n_per_season: int = 120, seasons=("1819", "1920", "2021", "2122"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, s in enumerate(seasons):
        x = rng.normal(size=n_per_season)
        shock = -0.4 if s == "1920" else 0.05 * i  # choque de nível numa temporada (pandemia)
        rows.append(pd.DataFrame({
            "season": s, "x": x, "xg_xa_p90": rng.gamma(2, 0.1, n_per_season), "age": rng.uniform(18, 34, n_per_season),
            "position_group": rng.choice(["DEF", "MEI", "ATA"], n_per_season),
            "age_band": rng.choice(["16–23", "24–28"], n_per_season),
            "market_log_return_lag1_T": shock,
            "log_return_T1": 0.5 * x + shock + rng.normal(0, 0.2, n_per_season),
        }))
    return pd.concat(rows, ignore_index=True)


def test_drift_adjusted_learns_relative_return_and_adds_forecast():
    d = _panel()
    inner = md._ridge({"alpha": 1.0}, ["x"], [])
    model = md.DriftAdjusted(inner, mode="mean").fit(d, d["log_return_T1"].to_numpy())
    assert model.season_drift_["1920"] < -0.3  # o choque fica no drift, não no modelo
    test = d[d["season"] == "2122"]
    np.testing.assert_allclose(model.drift_forecast(test), model.mean_drift_)
    regime = md.DriftAdjusted(inner, mode="regime", horizon=2).fit(d, d["log_return_T1"].to_numpy())
    np.testing.assert_allclose(regime.drift_forecast(test), 2 * test["market_log_return_lag1_T"])
    assert np.corrcoef(model.predict(test), test["x"])[0, 1] > 0.9


def test_build_model_wraps_excess_target_only_for_regression():
    task = md.Task(1, "global", "ridge", ("x",), (), target_mode="excess_regime")
    model = md.build_model(md.model_specs(1, 1)["ridge"], {"alpha": 1.0}, task)
    assert isinstance(model, md.DriftAdjusted) and model.mode == "regime"
    clf = md.Task(1, "global", "logit", ("x",), (), kind="clf", target_mode="excess_regime")
    assert not isinstance(md.build_model(md.model_specs(1, 1)["logit"], {"C": 1.0}, clf), md.DriftAdjusted)
    assert task.key() != md.Task(1, "global", "ridge", ("x",), ()).key()


def test_interaction_constraints_never_pair_two_performance_metrics():
    d = _panel()
    d["key_passes_p90"] = d["xg_xa_p90"] * 2
    numeric = ["x", "age", "xg_xa_p90", "key_passes_p90"]
    model = md._xgb_ic({"max_depth": 3}, numeric, ["position_group"])
    pre = md.Preprocessor(numeric, ["position_group"]).fit(d)
    est = model._prepare(model.est, pre)
    sets = json.loads(est.get_params()["interaction_constraints"])
    perf = {pre.feature_names_.index(c) for c in ("xg_xa_p90", "key_passes_p90")}
    assert all(len(perf & set(s)) <= 1 for s in sets)
    hubs = {i for i, n in enumerate(pre.feature_names_) if n.split("=")[0] in md.INTERACTION_HUBS}
    assert all(hubs <= set(s) for s in sets[1:])
    model.fit(d, d["log_return_T1"].to_numpy())  # o XGBoost aceita as restrições


def test_interaction_preprocessor_adds_cell_products():
    d = _panel()
    d["value_gap_T"] = d["x"]
    pre = md.InteractionPreprocessor(["x", "xg_xa_p90", "value_gap_T"], [], scale=True).fit(d)
    X = pre.transform(d)
    n_cells = d["age_band"].nunique() * d["position_group"].nunique() - 1
    assert X.shape[1] == 3 + 2 * n_cells == len(pre.feature_names_)


def test_classifier_and_multiquantile_tasks(tmp_path):
    d = _panel(80)
    for c in ("has_target_T1", "delta_v_pct_T1", "delta_v_T1", "value_T", "fbref_player_id", "fbref_team_id", "category"):
        d[c] = 1.0 if c != "category" else "C1"
    folds = [md.Fold(("1819", "1920"), "2021", ()), md.Fold(("1819", "1920", "2021"), "2122", ())]
    specs = md.model_specs(1, 1)
    clf = md.Task(1, "global", "logit", ("x",), (), kind="clf", hurdle=0.222)
    res = md.run_task(clf, d, folds, folds, specs, tmp_path, fixed_params=md.DEFAULT_PARAMS["logit"])
    assert res["predictions"]["y_pred"].between(0, 1).all()
    mq = md.Task(1, "global", "cat_mq", ("x",), (), kind="mq")
    res = md.run_task(mq, d, folds, folds, specs, tmp_path, fixed_params={"depth": 3})
    pred = res["predictions"]
    cols = [md.mq_column(q) for q in md.MQ_LEVELS]
    assert (np.diff(pred[cols].to_numpy(), axis=1) >= 0).all()  # quantis ordenados
    assert np.corrcoef(pred["y_pred"], d.loc[d["season"].isin(["2021", "2122"]), "x"])[0, 1] > 0.8


def test_multiquantile_mean_matches_lognormal_expectation():
    mu, sigma = -0.1, 0.5
    row = {md.mq_column(q): mu + sigma * stats.norm.ppf(q) for q in md.MQ_LEVELS}
    est = md.multiquantile_mean(pd.DataFrame([row])).iloc[0]
    exact = (np.exp(mu + sigma ** 2 / 2) - 1) * 100
    assert abs(est - exact) < 1.5  # p.p.; a regra do ponto médio perde só a extremidade das caudas


def _fold_preds(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for fold in range(1, 6):
        n = 400
        value = np.exp(rng.normal(15, 1, n))
        y_pred = rng.normal(0, 0.3, n)
        # variância residual cresce com a previsão: a cauda direita depende de x
        y_true = y_pred + rng.normal(0, 0.2 + 0.6 * (y_pred > 0.3), n)
        frames.append(pd.DataFrame({"fold": fold, "season": str(fold), "y_pred": y_pred, "y_true": y_true,
                                    "value_T": value, "age_band": rng.choice(["16–23", "24–28"], n),
                                    "dv_pct_true": np.expm1(y_true) * 100}))
    return pd.concat(frames, ignore_index=True)


def test_hetero_smearing_fixes_top_decile_better_than_duan():
    pred = _fold_preds()
    duan = md.magnitude_layer(pred, method="duan")
    hetero = md.magnitude_layer(pred, method="hetero")
    late = lambda m: m[m["fold"] >= 4]  # noqa: E731
    err = {k: md.decile_calibration_error(late(m), "dv_pct_pred_mean").query("decil == 10")["erro_abs"].mean()
           for k, m in (("duan", duan), ("hetero", hetero))}
    assert err["hetero"] < err["duan"]
    assert (duan.loc[duan["fold"] <= 2, "smearing"] == 1.0).all()  # sem histórico: identidade


def test_two_part_magnitude_uses_breakout_probability():
    pred = _fold_preds()
    pred["p_breakout"] = (pred["y_pred"] > 0.3).astype(float) * 0.5
    out = md.two_part_magnitude(pred)
    late = pred["fold"] >= 3
    assert out[late & (pred["p_breakout"] > 0)].mean() > out[late & (pred["p_breakout"] == 0)].mean()


def test_two_way_paired_test_reduces_to_one_way_with_same_clusters():
    rng = np.random.default_rng(5)
    n = 600
    y = rng.normal(size=n)
    a, b = y + rng.normal(0, 0.5, n), y + rng.normal(0, 0.6, n)
    club = rng.integers(0, 60, n)
    player = rng.integers(0, 200, n)
    one = md.paired_loss_test(y, a, b, club)
    same = md.paired_loss_test(y, a, b, club, clusters2=club)
    assert same["se"] == pytest.approx(one["se"])
    two = md.paired_loss_test(y, a, b, club, clusters2=player)
    assert two["clusters"] == 60 and two["se"] > 0
