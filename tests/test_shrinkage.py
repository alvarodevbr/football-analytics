import numpy as np
import pandas as pd

from football_analytics import features as ft


def _players(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_rate = rng.gamma(4.0, 0.1, n)            # taxa verdadeira por 90 (média 0,4)
    exposure = rng.uniform(1, 38, n)              # jogos de 90 minutos
    exposure[:20] = 1.0                           # 20 atletas com só 90 minutos
    counts = rng.poisson(true_rate * exposure)
    return pd.DataFrame({"count": counts, "exposure": exposure, "true": true_rate, "group": "g"})


def test_shrink_counts_pulls_low_exposure_toward_group_mean():
    d = _players()
    post = ft.shrink_counts(d["count"], d["exposure"], d["group"])
    raw = d["count"] / d["exposure"]
    mu = d["count"].sum() / d["exposure"].sum()
    low, high = d.index[:20], d.index[d["exposure"] > 30]
    # pouca exposição: perto da média do grupo; muita exposição: perto da taxa observada
    assert (post[low] - mu).abs().mean() < (raw[low] - mu).abs().mean() * 0.6
    assert (post[high] - raw[high]).abs().mean() < 0.05
    # o encolhimento reduz o erro em relação à taxa verdadeira
    assert ((post - d["true"]) ** 2).mean() < ((raw - d["true"]) ** 2).mean()


def test_shrink_counts_keeps_missing_exposure_missing():
    post = ft.shrink_counts(pd.Series([1.0, 2.0, 3.0, np.nan, 1.0]), pd.Series([1.0, 2.0, 0.0, 2.0, 3.0]),
                            pd.Series(["g"] * 5))
    assert np.isnan(post[2]) and np.isnan(post[3]) and post.notna().sum() == 3


def test_shrink_proportions_between_raw_and_prior():
    rng = np.random.default_rng(1)
    p_true = rng.beta(30, 10, 300)
    attempts = pd.Series(rng.integers(0, 400, 300)).astype(float)
    attempts[:5] = 0
    succ = pd.Series(rng.binomial(attempts.astype(int), p_true)).astype(float)
    post = ft.shrink_proportions(succ, attempts, pd.Series(["g"] * 300))
    pbar = succ.sum() / attempts.sum()
    assert np.allclose(post[:5], pbar)                          # sem tentativas: média do grupo
    raw = succ / attempts.where(attempts > 0)
    ok = attempts > 0
    lo, hi = np.minimum(raw[ok], pbar), np.maximum(raw[ok], pbar)
    assert ((post[ok] >= lo - 1e-12) & (post[ok] <= hi + 1e-12)).all()


def test_shrink_means_weights_by_precision():
    rng = np.random.default_rng(2)
    theta = rng.normal(6.8, 0.3, 500)
    w = pd.Series(rng.uniform(1, 38, 500))
    x = pd.Series(theta + rng.normal(0, 1.0, 500) / np.sqrt(w))
    post = ft.shrink_means(x, w, pd.Series(["g"] * 500))
    assert ((post - theta) ** 2).mean() < ((x - theta) ** 2).mean()
    mu = (w * x).sum() / w.sum()
    small, large = w < 3, w > 30
    assert (post[small] - mu).abs().mean() < (x[small] - mu).abs().mean()
    assert (post[large] - x[large]).abs().mean() < (post[small] - x[small]).abs().mean()


def test_attempts_from_published_percentage():
    n = ft._attempts_from_pct(pd.Series([6.0, 0.0, 3.0, 2.0]), pd.Series([60.0, 0.0, np.nan, 0.0]))
    assert n[0] == 10 and n[1] == 0 and np.isnan(n[2]) and np.isnan(n[3])
