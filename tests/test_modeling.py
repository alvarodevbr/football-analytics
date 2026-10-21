import numpy as np
import pandas as pd

from football_analytics import modeling as md
from football_analytics.validation import expanding_window_folds


def _panel(n_per_season=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in ["1920", "2021", "2122", "2223", "2324"]:
        for i in range(n_per_season):
            x = rng.normal()
            rows.append({"season": s, "fbref_player_id": f"{s}_{i}", "category": "C1" if i % 2 else "C2",
                         "position_group": "DEF" if i % 2 else "MEI", "league": "L1" if i % 3 else "L2",
                         "x": x, "z": np.nan if i % 7 == 0 else rng.normal(), "has_target_T1": 1.0,
                         "log_return_T1": 0.5 * x + rng.normal(scale=0.1), "value_T": 1e6})
    df = pd.DataFrame(rows)
    df["delta_v_pct_T1"] = np.expm1(df["log_return_T1"]) * 100
    df["delta_v_T1"] = df["value_T"] * np.expm1(df["log_return_T1"])
    return df


def test_preprocessor_group_median_imputation_and_dummies():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan], "g": ["A", "A", "B", "B"], "c": ["x", "y", "x", "y"]})
    X.loc[2, "a"] = 5.0
    pre = md.Preprocessor(["a"], ["c"], group_col="g").fit(X)
    out = pre.transform(X)
    assert out[1, 0] == 1.0 and out[3, 0] == 5.0          # mediana do próprio grupo
    assert out.shape == (4, 2) and list(out[:, 1]) == [0, 1, 0, 1]
    assert pre.feature_names_ == ["a", "c=y"]


def test_preprocessor_unseen_group_falls_back_to_global_median():
    train = pd.DataFrame({"a": [1.0, 3.0], "g": ["A", "A"]})
    test = pd.DataFrame({"a": [np.nan], "g": ["Z"]})
    assert md.Preprocessor(["a"], group_col="g").fit(train).transform(test)[0, 0] == 2.0


def test_median_baseline_by_group():
    X = pd.DataFrame({"category": ["C1", "C1", "C2"]})
    m = md.MedianBaseline("category").fit(X, np.array([1.0, 3.0, 10.0]))
    assert list(m.predict(pd.DataFrame({"category": ["C1", "C2", "C9"]}))) == [2.0, 10.0, 3.0]


def test_metrics_scales():
    pred = pd.DataFrame({"y_true": np.log([1.5, 0.5]), "y_pred": np.log([1.5, 0.5]),
                         "dv_pct_true": [50.0, -50.0], "dv_true": [5e5, -5e5], "value_T": [1e6, 1e6]})
    m = md.regression_metrics(pred)
    assert np.isclose(m["RMSE_log"], 0) and np.isclose(m["RMSE_pct"], 0) and np.isclose(m["RMSE_eur"], 0)


def test_run_task_tuning_never_sees_holdout(tmp_path):
    panel = _panel()
    folds = expanding_window_folds(["1920", "2021", "2122", "2223", "2324"], 1)
    specs = md.model_specs(n_calls_trees=6, n_calls_ridge=6)
    seen = []
    original = md.fit_predict

    def spy(spec, params, task, train, test):
        seen.append(test["season"].iloc[0])
        return original(spec, params, task, train, test)

    md.fit_predict = spy
    try:
        task = md.Task(1, "global", "ridge", ("x", "z"), ("league",))
        res = md.run_task(task, panel, folds, folds[:-1], specs, tmp_path)
    finally:
        md.fit_predict = original
    n_tuning = specs["ridge"].n_calls * 3
    assert "2324" not in seen[:n_tuning]                   # otimização só nas dobras de validação
    assert set(res["predictions"]["season"]) == {"2021", "2122", "2223", "2324"}
    assert res["predictions"].loc[res["predictions"]["is_holdout"], "season"].eq("2324").all()
    assert md.regression_metrics(res["predictions"])["R2_log"] > 0.8
    assert (tmp_path / f"{task.key()}.joblib").exists()   # cache para retomar


def test_magnitude_layer_uses_only_previous_folds_and_corrects_mean():
    rng = np.random.default_rng(0)
    rows = []
    for fold in (1, 2, 3):
        y_hat = rng.normal(size=400)
        y = y_hat + rng.normal(scale=0.6, size=400)          # ruído simétrico em log: média da razão > mediana
        rows.append(pd.DataFrame({"fold": fold, "y_pred": y_hat, "y_true": y, "age_band": "24–28"}))
    out = md.magnitude_layer(pd.concat(rows, ignore_index=True))
    f1 = out[out["fold"] == 1]
    assert np.allclose(f1["smearing"], 1.0) and np.allclose(f1["y_pred_cal"], f1["y_pred"])   # sem histórico
    later = out[out["fold"] == 3]["smearing"].iloc[0]
    assert abs(later - np.exp(0.6 ** 2 / 2)) < 0.05        # fator de Duan ≈ exp(σ²/2)
    assert (out["dv_pct_pred_mean"] >= out["dv_pct_pred_median"] - 1e-9).all()


def test_interval_metrics_coverage():
    pred = pd.DataFrame({"y_true": [0.0, 1.0, -1.0, 0.5], "q10": [-0.5] * 4, "q90": [0.6] * 4})
    m = md.interval_metrics(pred)
    assert m["cobertura"] == 0.5 and m["acima_do_superior"] == 0.25 and m["abaixo_do_inferior"] == 0.25


def test_conformal_interval_reaches_nominal_coverage():
    rng = np.random.default_rng(1)
    frames = []
    for fold in (1, 2, 3):
        y = rng.normal(size=2000)
        frames.append(pd.DataFrame({"fold": fold, "y_true": y, "q10": -0.5, "q90": 0.5}))  # intervalo estreito demais
    out = md.conformal_interval(pd.concat(frames, ignore_index=True))
    f3 = out[out["fold"] == 3]
    cov = ((f3["y_true"] >= f3["q10_conf"]) & (f3["y_true"] <= f3["q90_conf"])).mean()
    assert abs(cov - 0.8) < 0.03 and (out.loc[out["fold"] == 1, "conformal_adj"] == 0).all()
