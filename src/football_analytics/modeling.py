"""Treinamento e avaliação: pré-processamento dentro da dobra, algoritmos, otimização bayesiana
(processo gaussiano) sobre a janela expandida e métricas nas três escalas do alvo.

Convenções
* O alvo de treino é o log-retorno ln(V_{T+k}/V_T); ΔV% e ΔV (€) são derivados da previsão.
* Imputação (mediana por posição), winsorização e escalonamento são ajustados só com o treino de
  cada dobra (especificação, §6.6, passos 3–4).
* A otimização usa apenas as dobras de validação; a última dobra (teste em 2023/24) nunca é vista
  durante a escolha de hiperparâmetros e serve de teste final.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from .logging_utils import get_logger
from .validation import Fold

log = get_logger("modeling")

# Vetor compacto do OLS: um representante por bloco de alta correlação (notebook 03, Figura 3) e
# os controles da especificação. O Ridge e as árvores recebem o vetor completo.
OLS_FEATURES: tuple[str, ...] = (
    "age", "age_sq", "height_cm", "has_eu_passport", "is_domestic", "tenure_years_T", "is_on_loan_T",
    "log_value_T", "log_club_value_T", "team_points_per_game", "market_log_return_lag1_T",
    "minutes_share_pct", "starts_share_pct", "sofascore_rating", "goals_assists_p90", "xg_xa_p90",
    "key_passes_p90", "dribbles_p90", "tackles_interceptions_p90", "aerial_duels_won_p90",
    "pass_accuracy_pct", "log_return_lag1",
)


# --------------------------------------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------------------------------------
class Preprocessor(BaseEstimator, TransformerMixin):
    """Seleciona colunas, imputa a mediana do treino por grupo (posição), winsoriza e escala
    (opcionais) e codifica categóricas em *dummies* com categorias fixadas no treino."""

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str] = (), group_col: str | None = "position_group",
                 winsorize: tuple[float, float] | None = None, scale: bool = False):
        self.numeric = numeric
        self.categorical = categorical
        self.group_col = group_col
        self.winsorize = winsorize
        self.scale = scale

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Preprocessor":
        num = X[list(self.numeric)].astype(float)
        self.global_median_ = num.median()
        self.group_median_ = (num.groupby(X[self.group_col]).median()
                              if self.group_col and self.group_col in X else None)
        filled = self._impute(X, num)
        if self.winsorize:
            lo, hi = self.winsorize
            self.lower_, self.upper_ = filled.quantile(lo), filled.quantile(hi)
            filled = filled.clip(self.lower_, self.upper_, axis=1)
        if self.scale:
            self.center_ = filled.median()
            iqr = filled.quantile(0.75) - filled.quantile(0.25)
            std = filled.std().replace(0, 1.0)
            self.spread_ = iqr.where(iqr > 0, std).fillna(1.0)
        self.levels_ = {c: sorted(X[c].dropna().astype(str).unique()) for c in self.categorical}
        self.feature_names_ = list(self.numeric) + [f"{c}={v}" for c in self.categorical for v in self.levels_[c][1:]]
        return self

    def _impute(self, X: pd.DataFrame, num: pd.DataFrame) -> pd.DataFrame:
        if self.group_median_ is not None:
            by_group = self.group_median_.reindex(X[self.group_col].to_numpy())
            by_group.index = num.index
            num = num.fillna(by_group)
        return num.fillna(self.global_median_).fillna(0.0)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = self._impute(X, X[list(self.numeric)].astype(float))
        if self.winsorize:
            num = num.clip(self.lower_, self.upper_, axis=1)
        if self.scale:
            num = (num - self.center_) / self.spread_
        parts = [num.to_numpy(dtype=float)]
        for c in self.categorical:  # 1ª categoria como referência
            values = X[c].astype(str).to_numpy()
            parts += [(values == v).astype(float)[:, None] for v in self.levels_[c][1:]]
        return np.hstack(parts)


class Pipeline(BaseEstimator, RegressorMixin):
    """Pré-processador + estimador, ajustados juntos dentro da dobra."""

    def __init__(self, pre: Preprocessor, est: Any):
        self.pre = pre
        self.est = est

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Pipeline":
        self.pre_ = clone(self.pre).fit(X)
        self.est_ = clone(self.est).fit(self.pre_.transform(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est_.predict(self.pre_.transform(X))


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Prevê a mediana do treino (por grupo, se houver): a referência ingênua de reversão."""

    def __init__(self, group_col: str | None = None):
        self.group_col = group_col

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MedianBaseline":
        y = pd.Series(np.asarray(y), index=X.index)
        self.global_ = float(y.median())
        self.by_group_ = y.groupby(X[self.group_col]).median() if self.group_col else None
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.by_group_ is None:
            return np.full(len(X), self.global_)
        return X[self.group_col].map(self.by_group_).fillna(self.global_).to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------
# Algoritmos e espaços de busca
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    build: Callable[[dict[str, Any], Sequence[str], Sequence[str]], Any]
    space: list[Any] = field(default_factory=list)       # dimensões do skopt
    param_names: list[str] = field(default_factory=list)
    n_calls: int = 0                                     # 0 = sem otimização


def _xgb(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from xgboost import XGBRegressor

    est = XGBRegressor(tree_method="hist", max_bin=64, n_jobs=1, random_state=42, verbosity=0, **params)
    return Pipeline(Preprocessor(numeric, categorical), est)


def _cat(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from catboost import CatBoostRegressor

    # rsm: amostra de 50% das variáveis por divisão (custo de CPU; ver notebook 04)
    est = CatBoostRegressor(thread_count=1, random_seed=42, verbose=0, allow_writing_files=False,
                            border_count=64, rsm=0.5, **params)
    return Pipeline(Preprocessor(numeric, categorical), est)


def _rf(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    params = dict(params)
    mf = params.pop("max_features")
    params["max_features"] = mf if mf in ("sqrt", "log2") else float(mf)
    est = RandomForestRegressor(n_jobs=1, random_state=42, **params)
    return Pipeline(Preprocessor(numeric, categorical), est)


def _ridge(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline(Preprocessor(numeric, categorical, winsorize=(0.01, 0.99), scale=True), Ridge(**params))


def _ols(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    cols = [c for c in OLS_FEATURES if c in numeric]
    return Pipeline(Preprocessor(cols, categorical, winsorize=(0.01, 0.99), scale=True), LinearRegression())


def _median(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> MedianBaseline:
    return MedianBaseline()  # o grupo (categoria, no escopo global) é definido em `fit_predict`


def model_specs(n_calls_trees: int, n_calls_ridge: int) -> dict[str, ModelSpec]:
    """Espaços de busca da especificação (§6.6), com cortes de custo computacional documentados no
    notebook 04: XGBoost e CatBoost com até 800 árvores/iterações, CatBoost com profundidade até 8 e
    Random Forest com até 500 árvores e max_features em {sqrt, log2, 0,3}."""
    from skopt.space import Categorical, Integer, Real

    return {
        "median": ModelSpec("median", "Mediana do treino (ingênuo)", _median),
        "ols": ModelSpec("ols", "OLS", _ols),
        "ridge": ModelSpec("ridge", "Ridge", _ridge, [Real(1e-2, 1e4, prior="log-uniform")], ["alpha"], n_calls_ridge),
        "rf": ModelSpec("rf", "Random Forest", _rf, [
            Integer(100, 500), Integer(5, 25), Integer(2, 10), Integer(1, 5),
            Categorical(["sqrt", "log2", "0.3"])],
            ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"], n_calls_trees),
        "xgb": ModelSpec("xgb", "XGBoost", _xgb, [
            Integer(3, 10), Real(0.01, 0.30, prior="log-uniform"), Integer(100, 800), Real(0.6, 1.0),
            Real(0.6, 1.0), Real(0.0, 5.0), Real(1e-3, 10.0, prior="log-uniform"), Real(1e-3, 10.0, prior="log-uniform")],
            ["max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree", "gamma", "reg_alpha",
             "reg_lambda"], n_calls_trees),
        "cat": ModelSpec("cat", "CatBoost", _cat, [
            Integer(4, 8), Real(0.01, 0.30, prior="log-uniform"), Real(1.0, 10.0), Integer(200, 800)],
            ["depth", "learning_rate", "l2_leaf_reg", "iterations"], n_calls_trees),
    }


# --------------------------------------------------------------------------------------------
# Dados por dobra e métricas
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """Uma combinação horizonte × escopo (global ou categoria) × algoritmo."""

    horizon: int
    scope: str                       # "global" ou "C1".."C12"
    model: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    extra_label: str = ""            # ex.: "lookahead" para o modelo de robustez

    @property
    def target(self) -> str:
        return f"log_return_T{self.horizon}"

    def key(self) -> str:
        payload = json.dumps([self.horizon, self.scope, self.model, self.numeric, self.categorical, self.extra_label])
        return f"T{self.horizon}_{self.scope}_{self.model}{'_' + self.extra_label if self.extra_label else ''}_" \
               + hashlib.md5(payload.encode()).hexdigest()[:8]


def task_data(panel: pd.DataFrame, task: Task) -> pd.DataFrame:
    d = panel[panel[f"has_target_T{task.horizon}"] == 1]
    if task.scope != "global":
        d = d[d["category"] == task.scope]
    return d


def fold_split(d: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    return d[d["season"].isin(fold.train)], d[d["season"] == fold.test]


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan


def regression_metrics(pred: pd.DataFrame, pct_col: str | None = None) -> pd.Series:
    """Métricas nas três escalas: log-retorno (treino), ΔV% e ΔV (€); mais ordenação e decil.

    `pct_col` escolhe a previsão de ΔV% (por padrão, a mediana exp(ŷ) − 1; a camada de magnitude
    produz também a esperança corrigida por *smearing*). ΔV em € é sempre V_T × ΔV%.
    """
    y, p = pred["y_true"].to_numpy(), pred["y_pred"].to_numpy()
    pct_t = pred["dv_pct_true"].to_numpy()
    pct_p = pred[pct_col].to_numpy() if pct_col else np.expm1(p) * 100
    eur_t, eur_p = pred["dv_true"].to_numpy(), pred["value_T"].to_numpy() * pct_p / 100
    out = {
        "n": len(pred),
        "RMSE_log": rmse(y, p), "MAE_log": float(np.mean(np.abs(y - p))), "R2_log": r2(y, p),
        "RMSE_pct": rmse(pct_t, pct_p), "MAE_pct": float(np.mean(np.abs(pct_t - pct_p))),
        "MedAE_pct": float(np.median(np.abs(pct_t - pct_p))), "R2_pct": r2(pct_t, pct_p),
        "RMSE_eur": rmse(eur_t, eur_p), "MAE_eur": float(np.mean(np.abs(eur_t - eur_p))), "R2_eur": r2(eur_t, eur_p),
        "spearman": float(stats.spearmanr(y, p).statistic) if len(pred) > 2 and np.std(p) > 0 else np.nan,
        # magnitude: inclinação de y ~ ŷ (1 = sem encolhimento) e erro entre os que se valorizaram
        "calib_slope": float(np.polyfit(p, y, 1)[0]) if len(pred) > 2 and np.std(p) > 0 else np.nan,
        "MAE_pct_valorizados": float(np.mean(np.abs(pct_t - pct_p)[y > 0])) if (y > 0).any() else np.nan,
    }
    # decisão buy-low: dos 10% com maior valorização prevista, quantos de fato se valorizaram
    out["base_hit_rate"] = float((pred["y_true"] > 0).mean()) if len(pred) else np.nan
    out["top10_hit_rate"] = out["top10_median_dv_pct"] = np.nan
    if len(pred) >= 20:
        top = pred.assign(_p=p).nlargest(max(1, len(pred) // 10), "_p")
        out["top10_hit_rate"] = float((top["y_true"] > 0).mean())
        out["top10_median_dv_pct"] = float(top["dv_pct_true"].median())
    return pd.Series(out)


def magnitude_layer(pred: pd.DataFrame, group_col: str = "age_band", min_prev: int = 200) -> pd.DataFrame:
    """Converte o log-retorno previsto em magnitudes de ΔV%, sem olhar o futuro.

    Para cada dobra, usa só as previsões fora da amostra das dobras **anteriores**:
    1. recalibração linear ŷ' = a + b·ŷ (corrige encolhimento ou excesso de dispersão);
    2. *smearing* de Duan (1983) por grupo, f_g = média de exp(y − ŷ'), porque exp(ŷ') estima a
       **mediana** da razão V_{T+k}/V_T, não sua média.

    Devolve `dv_pct_pred_median` = exp(ŷ') − 1 (ponto para erros absolutos) e `dv_pct_pred_mean` =
    f_g·exp(ŷ') − 1 (esperança, ponto para erros quadráticos e somas em €). Na 1ª dobra, sem
    histórico, a camada é a identidade.
    """
    out = []
    for fold in sorted(pred["fold"].unique()):
        cur = pred[pred["fold"] == fold].copy()
        prev = pred[pred["fold"] < fold]
        a, b, smear, overall = 0.0, 1.0, pd.Series(dtype=float), 1.0
        if len(prev) >= min_prev:
            b, a = np.polyfit(prev["y_pred"], prev["y_true"], 1)
            ratio = np.exp(prev["y_true"] - (a + b * prev["y_pred"]))
            smear, overall = ratio.groupby(prev[group_col]).mean(), float(ratio.mean())
        cur["y_pred_cal"] = a + b * cur["y_pred"]
        cur["smearing"] = cur[group_col].map(smear).fillna(overall).astype(float)
        cur["dv_pct_pred_median"] = np.expm1(cur["y_pred_cal"]) * 100
        cur["dv_pct_pred_mean"] = (np.exp(cur["y_pred_cal"]) * cur["smearing"] - 1) * 100
        out.append(cur)
    return pd.concat(out)


def conformal_interval(pred: pd.DataFrame, lo: str = "q10", hi: str = "q90", nominal: float = 0.8,
                       min_prev: int = 200) -> pd.DataFrame:
    """Regressão quantílica conformalizada (Romano et al., 2019), sequencial no tempo.

    Para cada dobra, o escore de conformidade max(q_lo − y, y − q_hi) das dobras **anteriores**
    define o ajuste que leva a cobertura empírica ao nível nominal. Na 1ª dobra, sem histórico,
    mantém o intervalo bruto.
    """
    out = []
    for fold in sorted(pred["fold"].unique()):
        cur = pred[pred["fold"] == fold].copy()
        prev = pred[pred["fold"] < fold]
        adj = 0.0
        if len(prev) >= min_prev:
            score = np.maximum(prev[lo] - prev["y_true"], prev["y_true"] - prev[hi])
            adj = float(np.quantile(score, min(1.0, nominal * (1 + 1 / len(prev)))))
        cur[f"{lo}_conf"], cur[f"{hi}_conf"], cur["conformal_adj"] = cur[lo] - adj, cur[hi] + adj, adj
        out.append(cur)
    return pd.concat(out)


def interval_metrics(pred: pd.DataFrame, lo: str = "q10", hi: str = "q90", nominal: float = 0.8) -> pd.Series:
    """Cobertura e largura do intervalo preditivo [lo, hi] em log-retorno e na escala de ΔV%."""
    y = pred["y_true"]
    inside = (y >= pred[lo]) & (y <= pred[hi])
    width_pct = (np.expm1(pred[hi]) - np.expm1(pred[lo])) * 100
    return pd.Series({"n": len(pred), "cobertura": float(inside.mean()), "cobertura_nominal": nominal,
                      "largura_mediana_pp": float(width_pct.median()),
                      "abaixo_do_inferior": float((y < pred[lo]).mean()), "acima_do_superior": float((y > pred[hi]).mean()),
                      "pinball_q10": float(np.mean(np.maximum(0.1 * (y - pred[lo]), 0.9 * (pred[lo] - y)))),
                      "pinball_q90": float(np.mean(np.maximum(0.9 * (y - pred[hi]), 0.1 * (pred[hi] - y))))})


# --------------------------------------------------------------------------------------------
# Otimização, avaliação e cache
# --------------------------------------------------------------------------------------------
def fit_predict(spec: ModelSpec, params: dict[str, Any], task: Task, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    if spec.key == "median":
        model = MedianBaseline(group_col="category" if task.scope == "global" else None)
    else:
        model = spec.build(params, task.numeric, task.categorical)
    model.fit(train, train[task.target].to_numpy())
    return model.predict(test)


def cv_rmse(spec: ModelSpec, params: dict[str, Any], task: Task, d: pd.DataFrame, folds: Sequence[Fold]) -> float:
    scores = []
    for fold in folds:
        train, test = fold_split(d, fold)
        if len(train) < 10 or test.empty:
            continue
        scores.append(rmse(test[task.target], fit_predict(spec, params, task, train, test)))
    return float(np.mean(scores)) if scores else np.inf


def tune(spec: ModelSpec, task: Task, d: pd.DataFrame, folds: Sequence[Fold], seed: int = 42) -> tuple[dict[str, Any], list[float]]:
    """gp_minimize (processo gaussiano, aquisição gp_hedge) da RMSE média nas dobras de validação."""
    if not spec.space:
        return {}, []
    from skopt import gp_minimize

    def objective(x: list[Any]) -> float:
        return cv_rmse(spec, dict(zip(spec.param_names, x)), task, d, folds)

    n_init = max(5, spec.n_calls // 2)
    res = gp_minimize(objective, spec.space, n_calls=spec.n_calls, n_initial_points=n_init, random_state=seed)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in zip(spec.param_names, res.x)}
    return best, [float(v) for v in res.func_vals]


def run_task(task: Task, panel: pd.DataFrame, folds: Sequence[Fold], tuning_folds: Sequence[Fold],
             specs: dict[str, ModelSpec], cache_dir: Path, fixed_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Otimiza (dobras de validação), avalia em todas as dobras e grava em cache.

    Com `fixed_params`, pula a otimização (usado no modelo de robustez, que reaproveita os
    hiperparâmetros do modelo principal para isolar o efeito das variáveis adicionais).
    """
    path = cache_dir / f"{task.key()}.joblib"
    if path.exists():
        return joblib.load(path)
    t0 = time.time()
    spec = specs[task.model]
    d = task_data(panel, task)
    if fixed_params is not None:
        params, history = dict(fixed_params), []
    else:
        params, history = tune(spec, task, d, tuning_folds)
    preds = []
    for i, fold in enumerate(folds, 1):
        train, test = fold_split(d, fold)
        if len(train) < 10 or test.empty:
            continue
        p = fit_predict(spec, params, task, train, test)
        h = task.horizon
        preds.append(pd.DataFrame({
            "season": test["season"].to_numpy(), "fbref_player_id": test["fbref_player_id"].to_numpy(),
            "category": test["category"].to_numpy(), "fold": i, "is_holdout": i == len(folds),
            "y_true": test[task.target].to_numpy(), "y_pred": p,
            "dv_pct_true": test[f"delta_v_pct_T{h}"].to_numpy(), "dv_true": test[f"delta_v_T{h}"].to_numpy(),
            "value_T": test["value_T"].to_numpy(),
        }))
    result = {"task": task, "params": params, "history": history,
              "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
              "seconds": time.time() - t0, "n_rows": len(d)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    log.info("tarefa concluída", extra={"key": task.key(), "seconds": round(result["seconds"], 1)})
    return result


def fit_final(spec: ModelSpec, params: dict[str, Any], task: Task, panel: pd.DataFrame) -> Any:
    """Modelo final: hiperparâmetros otimizados, ajustado em todas as temporadas com alvo."""
    d = task_data(panel, task)
    if spec.key == "median":
        model = MedianBaseline(group_col="category" if task.scope == "global" else None)
    else:
        model = spec.build(params, task.numeric, task.categorical)
    return model.fit(d, d[task.target].to_numpy())
