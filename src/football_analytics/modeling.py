"""Treinamento e avaliação: pré-processamento dentro da dobra, algoritmos, otimização bayesiana
(processo gaussiano) sobre a janela expandida, métricas nas três escalas do alvo e inferência
(bootstrap por cluster, teste pareado de perdas e conjunto de confiança de modelos).

Convenções
* O alvo de treino é o log-retorno ln(V_{T+k}/V_T); ΔV% e ΔV (€) são derivados da previsão.
* Imputação (mediana por posição), winsorização e escalonamento são ajustados só com o treino de
  cada dobra (especificação, §6.6, passos 3–4).
* A otimização usa apenas as dobras de validação; a última dobra (teste final de cada horizonte)
  nunca é vista durante a escolha de hiperparâmetros.
* XGBoost e CatBoost têm taxa de aprendizado fixa, e o número de árvores sai de parada antecipada
  temporal: a última temporada do treino é a validação interna.
* As previsões cobrem também as linhas de teste sem alvo observado (atletas que saíram da base),
  para o backtest por intenção de tratar; as métricas ignoram essas linhas.
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


class EarlyStoppingPipeline(Pipeline):
    """Pré-processador + *booster* com o número de árvores escolhido por parada antecipada temporal.

    A última temporada do treino serve de validação interna, com purga de `purge` temporadas entre
    ela e o treino interno (1 no horizonte T+2, cujo alvo cruza a temporada seguinte). O modelo é
    reajustado em todo o treino com o melhor número de árvores. Sem temporadas ou linhas suficientes
    (a 1ª dobra treina com uma temporada só), usa `default_trees`.
    """

    def __init__(self, pre: Preprocessor, est: Any, kind: str, max_trees: int = 1000, patience: int = 40,
                 default_trees: int = 200, purge: int = 0, min_rows: int = 30, season_col: str = "season"):
        super().__init__(pre, est)
        self.kind = kind
        self.max_trees = max_trees
        self.patience = patience
        self.default_trees = default_trees
        self.purge = purge
        self.min_rows = min_rows
        self.season_col = season_col

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EarlyStoppingPipeline":
        y = np.asarray(y)
        seasons = X[self.season_col]
        ordered = sorted(seasons.unique())
        n_trees = self.default_trees
        if len(ordered) >= 2 + self.purge:
            inner = seasons.isin(ordered[:len(ordered) - 1 - self.purge]).to_numpy()
            val = (seasons == ordered[-1]).to_numpy()
            if inner.sum() >= self.min_rows and val.sum() >= self.min_rows:
                pre = clone(self.pre).fit(X[inner])
                n_trees = self._best_iteration(pre.transform(X[inner]), y[inner], pre.transform(X[val]), y[val],
                                               self._prepare(clone(self.est), pre))
        self.n_trees_ = int(n_trees)
        self.pre_ = clone(self.pre).fit(X)
        est = self._prepare(clone(self.est), self.pre_)
        self.est_ = self._with_trees(est, self.n_trees_).fit(self.pre_.transform(X), y)
        return self

    def _prepare(self, est: Any, pre: Preprocessor) -> Any:
        """Ajuste do estimador que depende das colunas do pré-processador (restrições de interação)."""
        return est

    def _best_iteration(self, Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray, yv: np.ndarray, est: Any) -> int:
        if self.kind == "xgb":
            est.set_params(n_estimators=self.max_trees, early_stopping_rounds=self.patience)
            est.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            return int(est.best_iteration) + 1
        est.set_params(iterations=self.max_trees, od_type="Iter", od_wait=self.patience)
        est.fit(Xt, yt, eval_set=(Xv, yv), use_best_model=False)
        return int(est.get_best_iteration()) + 1

    def _with_trees(self, est: Any, n_trees: int) -> Any:
        if self.kind == "xgb":
            return est.set_params(n_estimators=n_trees, early_stopping_rounds=None)
        return est.set_params(iterations=n_trees)


class _ProbaMixin:
    """`predict` devolve P(classe 1): os classificadores entram no mesmo encanamento dos regressores."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.est_.predict_proba(self.pre_.transform(X))[:, 1]


class ProbaPipeline(_ProbaMixin, Pipeline):
    pass


class ProbaEarlyStoppingPipeline(_ProbaMixin, EarlyStoppingPipeline):
    pass


# Grupos de desempenho e "eixos" das restrições de interação: cada métrica de desempenho só pode
# interagir com idade, faixa etária, posição e papel tático, nunca com outra métrica de desempenho.
PERFORMANCE_GROUPS: tuple[str, ...] = ("Ofensivo", "Criação e posse", "Defensivo", "Disciplina", "Avaliação agregada",
                                      "Forma intra-temporada")
INTERACTION_HUBS: tuple[str, ...] = ("age", "age_sq", "age_band", "position_group", "role")


def performance_columns(numeric: Sequence[str]) -> list[str]:
    from .feature_catalog import CATALOG

    return [c for c in numeric if c in CATALOG and CATALOG[c].group in PERFORMANCE_GROUPS]


class InteractionConstrainedPipeline(EarlyStoppingPipeline):
    """XGBoost com `interaction_constraints`: um conjunto com todas as variáveis que não são de
    desempenho (livres entre si) e, para cada métrica de desempenho, um conjunto {métrica, eixos}.
    Evita interações espúrias de 3 ou 4 ordens entre taxas ruidosas sem partir a amostra em células."""

    def __init__(self, pre: Preprocessor, est: Any, kind: str, perf_features: Sequence[str] = (),
                 hubs: Sequence[str] = INTERACTION_HUBS, max_trees: int = 1000, patience: int = 40,
                 default_trees: int = 200, purge: int = 0, min_rows: int = 30, season_col: str = "season"):
        super().__init__(pre, est, kind, max_trees, patience, default_trees, purge, min_rows, season_col)
        self.perf_features = perf_features
        self.hubs = hubs

    def _prepare(self, est: Any, pre: Preprocessor) -> Any:
        names = pre.feature_names_
        perf_set = set(self.perf_features)
        hub = [i for i, n in enumerate(names) if n.split("=")[0] in self.hubs]
        perf = [i for i, n in enumerate(names) if n in perf_set]
        rest = [i for i, n in enumerate(names) if n not in perf_set]
        sets = [rest] + [sorted({p, *hub}) for p in perf]
        # com matriz numpy, o XGBoost só aceita índices na forma de texto
        return est.set_params(interaction_constraints=json.dumps(sets))


# Compostos de desempenho com coeficiente variável por faixa etária × posição no Ridge com interações:
# produção ofensiva, criação, ações defensivas, nota, regularidade e gap de avaliação.
INTERACTION_COMPOSITES: tuple[str, ...] = (
    "xg_xa_p90", "key_passes_p90", "tackles_interceptions_p90", "sofascore_rating", "minutes_share_pct", "value_gap_T",
)


class InteractionPreprocessor(Preprocessor):
    """Preprocessor + produtos `composto × 1(célula)`, com célula = faixa etária × posição: um
    modelo de coeficientes variáveis, estimado na amostra inteira (as demais variáveis têm um
    coeficiente único), em vez de 12 modelos separados."""

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str] = (), group_col: str | None = "position_group",
                 winsorize: tuple[float, float] | None = None, scale: bool = False,
                 interact: Sequence[str] = INTERACTION_COMPOSITES, cells: Sequence[str] = ("age_band", "position_group")):
        super().__init__(numeric, categorical, group_col, winsorize, scale)
        self.interact = interact
        self.cells = cells

    def _cell(self, X: pd.DataFrame) -> np.ndarray:
        return X[list(self.cells)].astype(str).agg("|".join, axis=1).to_numpy()

    def fit(self, X: pd.DataFrame, y: Any = None) -> "InteractionPreprocessor":
        super().fit(X, y)
        self.interact_ = [c for c in self.interact if c in self.numeric]
        self.cell_levels_ = sorted(set(self._cell(X)))[1:]
        self.feature_names_ = self.feature_names_ + [f"{c}×{lvl}" for c in self.interact_ for lvl in self.cell_levels_]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        base = super().transform(X)
        cell = self._cell(X)
        idx = [list(self.numeric).index(c) for c in self.interact_]
        cols = [base[:, i] * (cell == lvl) for i in idx for lvl in self.cell_levels_]
        return np.hstack([base, np.column_stack(cols)]) if cols else base


class DriftAdjusted(BaseEstimator, RegressorMixin):
    """Treina no log-retorno em excesso ao *drift* de mercado da temporada e devolve a previsão nominal.

    O *drift* de cada temporada de treino é a média aparada (10%) do alvo nela: o modelo aprende só a
    valorização relativa, sem gastar capacidade com choques de nível como a revisão de abril de 2020.
    Na previsão, o *drift* da temporada de teste, desconhecido em T, é estimado por `mode`:
    * "mean": média dos *drifts* das temporadas de treino;
    * "regime": k × regime de mercado de T (log-retorno médio de T−1 para T, conhecido em T).
    """

    def __init__(self, inner: Any, mode: str = "mean", horizon: int = 1, season_col: str = "season",
                 regime_col: str = "market_log_return_lag1_T", trim: float = 0.1):
        self.inner = inner
        self.mode = mode
        self.horizon = horizon
        self.season_col = season_col
        self.regime_col = regime_col
        self.trim = trim

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DriftAdjusted":
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        self.season_drift_ = y.groupby(X[self.season_col]).agg(lambda v: stats.trim_mean(v, self.trim))
        self.mean_drift_ = float(self.season_drift_.mean())
        excess = y - X[self.season_col].map(self.season_drift_)
        self.inner_ = clone(self.inner).fit(X, excess.to_numpy())
        return self

    def drift_forecast(self, X: pd.DataFrame) -> np.ndarray:
        if self.mode == "regime":
            regime = X[self.regime_col].astype(float) * self.horizon
            return regime.fillna(self.mean_drift_).to_numpy()
        return np.full(len(X), self.mean_drift_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.inner_.predict(X) + self.drift_forecast(X)


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


# Versão da especificação dos modelos: entra na chave do cache, que se renova quando ela muda.
SPEC_VERSION = "v4"
LEARNING_RATE = 0.05  # fixa nos boosters; o número de árvores sai da parada antecipada


def _xgb(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from xgboost import XGBRegressor

    est = XGBRegressor(tree_method="hist", max_bin=64, n_jobs=1, random_state=42, verbosity=0,
                       learning_rate=LEARNING_RATE, **params)
    return EarlyStoppingPipeline(Preprocessor(numeric, categorical), est, kind="xgb")


def _cat(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from catboost import CatBoostRegressor

    # rsm: amostra de 50% das variáveis por divisão (custo de CPU; ver notebook 04)
    est = CatBoostRegressor(thread_count=1, random_seed=42, verbose=0, allow_writing_files=False,
                            border_count=64, rsm=0.5, learning_rate=LEARNING_RATE, **params)
    return EarlyStoppingPipeline(Preprocessor(numeric, categorical), est, kind="cat")


def _rf(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    params = dict(params)
    mf = params.pop("max_features")
    params["max_features"] = mf if mf in ("sqrt", "log2") else float(mf)
    est = RandomForestRegressor(n_estimators=150, n_jobs=1, random_state=42, **params)
    return Pipeline(Preprocessor(numeric, categorical), est)


def _ridge(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline(Preprocessor(numeric, categorical, winsorize=(0.01, 0.99), scale=True), Ridge(**params))


def _ols(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    cols = [c for c in OLS_FEATURES if c in numeric]
    return Pipeline(Preprocessor(cols, categorical, winsorize=(0.01, 0.99), scale=True), LinearRegression())


def _median(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> MedianBaseline:
    return MedianBaseline()  # o grupo (categoria, no escopo global) é definido em `fit_predict`


def _xgb_ic(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from xgboost import XGBRegressor

    est = XGBRegressor(tree_method="hist", max_bin=64, n_jobs=1, random_state=42, verbosity=0,
                       learning_rate=LEARNING_RATE, **params)
    return InteractionConstrainedPipeline(Preprocessor(numeric, categorical), est, kind="xgb",
                                          perf_features=tuple(performance_columns(numeric)))


def _ridge_ic(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    return Pipeline(InteractionPreprocessor(numeric, categorical, winsorize=(0.01, 0.99), scale=True), Ridge(**params))


# Níveis do CatBoost multiquantil: pontos médios de 20 faixas de probabilidade de 5%. A média de
# exp(q_τ) − 1 sobre eles aproxima a integral de exp(q_τ) − 1 em τ ∈ (0, 1) pela regra do ponto médio.
MQ_LEVELS: tuple[float, ...] = tuple(round(0.025 + 0.05 * i, 3) for i in range(20))


def _cat_mq(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from catboost import CatBoostRegressor

    alphas = ",".join(f"{q:g}" for q in MQ_LEVELS)
    est = CatBoostRegressor(thread_count=1, random_seed=42, verbose=0, allow_writing_files=False,
                            border_count=64, rsm=0.5, learning_rate=LEARNING_RATE,
                            loss_function=f"MultiQuantile:alpha={alphas}", **params)
    return EarlyStoppingPipeline(Preprocessor(numeric, categorical), est, kind="cat")


def _cat_clf(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from catboost import CatBoostClassifier

    est = CatBoostClassifier(thread_count=1, random_seed=42, verbose=0, allow_writing_files=False,
                             border_count=64, rsm=0.5, learning_rate=LEARNING_RATE, **params)
    return ProbaEarlyStoppingPipeline(Preprocessor(numeric, categorical), est, kind="cat")


def _xgb_clf(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from xgboost import XGBClassifier

    est = XGBClassifier(tree_method="hist", max_bin=64, n_jobs=1, random_state=42, verbosity=0,
                        learning_rate=LEARNING_RATE, eval_metric="logloss", **params)
    return ProbaEarlyStoppingPipeline(Preprocessor(numeric, categorical), est, kind="xgb")


def _logit(params: dict[str, Any], numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    from sklearn.linear_model import LogisticRegression

    return ProbaPipeline(Preprocessor(numeric, categorical, winsorize=(0.01, 0.99), scale=True),
                         LogisticRegression(max_iter=5000, **params))


def model_specs(n_calls_trees: int, n_calls_ridge: int) -> dict[str, ModelSpec]:
    """Espaços de busca centrados em regularização (v3).

    Na v2, os ótimos caíam na fronteira dos espaços da especificação (§6.6): poucas árvores,
    profundidade máxima, penalizações no limite. Na v3, a taxa de aprendizado é fixa, o número de
    árvores sai da parada antecipada e a busca cobre só a complexidade e a regularização, com
    intervalos que vão até penalizações fortes. Isso reduz a dimensão (de 8 para 5 no XGBoost e de
    4 para 3 no CatBoost) e torna o orçamento de avaliações compatível com o processador disponível.
    """
    from skopt.space import Categorical, Integer, Real

    return {
        "median": ModelSpec("median", "Mediana do treino (ingênuo)", _median),
        "ols": ModelSpec("ols", "OLS", _ols),
        "ridge": ModelSpec("ridge", "Ridge", _ridge, [Real(1e-2, 1e4, prior="log-uniform")], ["alpha"], n_calls_ridge),
        "rf": ModelSpec("rf", "Random Forest", _rf, [
            Integer(5, 25), Integer(1, 50, prior="log-uniform"), Categorical(["sqrt", "0.1", "0.2"])],
            ["max_depth", "min_samples_leaf", "max_features"], n_calls_trees),
        "xgb": ModelSpec("xgb", "XGBoost", _xgb, [
            Integer(2, 8), Real(1.0, 100.0, prior="log-uniform"), Real(0.1, 100.0, prior="log-uniform"),
            Real(0.5, 1.0), Real(0.3, 1.0)],
            ["max_depth", "min_child_weight", "reg_lambda", "subsample", "colsample_bytree"], n_calls_trees),
        "cat": ModelSpec("cat", "CatBoost", _cat, [
            Integer(3, 8), Real(1.0, 100.0, prior="log-uniform"), Real(0.0, 5.0)],
            ["depth", "l2_leaf_reg", "random_strength"], n_calls_trees),
        # v4: arquitetura global com interações restritas (alternativa à segmentação em células)
        "xgb_ic": ModelSpec("xgb_ic", "XGBoost com interações restritas", _xgb_ic, [
            Integer(2, 8), Real(1.0, 100.0, prior="log-uniform"), Real(0.1, 100.0, prior="log-uniform"),
            Real(0.5, 1.0), Real(0.3, 1.0)],
            ["max_depth", "min_child_weight", "reg_lambda", "subsample", "colsample_bytree"], n_calls_trees),
        "ridge_ic": ModelSpec("ridge_ic", "Ridge com coeficientes por faixa × posição", _ridge_ic,
                              [Real(1e-2, 1e4, prior="log-uniform")], ["alpha"], n_calls_ridge),
        # v4: distribuição (multiquantil) e decisão (classificadores), com hiperparâmetros fixos
        "cat_mq": ModelSpec("cat_mq", "CatBoost multiquantil", _cat_mq),
        "cat_clf": ModelSpec("cat_clf", "CatBoost (classificação)", _cat_clf),
        "xgb_clf": ModelSpec("xgb_clf", "XGBoost (classificação)", _xgb_clf),
        "logit": ModelSpec("logit", "Logística L2", _logit),
    }


# Hiperparâmetros-padrão, sem otimização: robustez que mede quanto o HPO otimiza as dobras de
# validação em que foi feito.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 10.0},
    "rf": {"max_depth": 20, "min_samples_leaf": 10, "max_features": "0.1"},
    "xgb": {"max_depth": 4, "min_child_weight": 10.0, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.7},
    "cat": {"depth": 6, "l2_leaf_reg": 10.0, "random_strength": 1.0},
    "xgb_ic": {"max_depth": 4, "min_child_weight": 10.0, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.7},
    "ridge_ic": {"alpha": 10.0},
    "cat_mq": {"depth": 6, "l2_leaf_reg": 10.0, "random_strength": 1.0},
    "cat_clf": {"depth": 6, "l2_leaf_reg": 10.0, "random_strength": 1.0},
    "xgb_clf": {"max_depth": 4, "min_child_weight": 10.0, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.7},
    "logit": {"C": 0.05},
}


# --------------------------------------------------------------------------------------------
# Dados por dobra e métricas
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """Uma combinação horizonte × escopo (global ou segmento) × algoritmo."""

    horizon: int
    scope: str                       # "global", "C1".."C12" ou "K1".."K9"
    model: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    extra_label: str = ""            # ex.: "lookahead" para o modelo de robustez
    scope_col: str = "category"      # coluna do segmento: "category" (4 × 3) ou "category_3x3"
    data_tag: str = ""               # hash do painel: dados novos geram chaves novas no cache
    version: str = SPEC_VERSION
    kind: str = "reg"                # "reg" (log-retorno), "mq" (multiquantil) ou "clf" (classificação)
    hurdle: float = 0.0              # classificação: rótulo 1(ΔV% > hurdle), ΔV% em fração
    target_mode: str = "nominal"     # "nominal", "excess_mean" ou "excess_regime" (ver `DriftAdjusted`)
    target_name: str = ""            # alvo alternativo (robustez); vazio = log-retorno de T+k

    @property
    def target(self) -> str:
        return self.target_name or f"log_return_T{self.horizon}"

    def key(self) -> str:
        payload = json.dumps([self.horizon, self.scope, self.model, self.numeric, self.categorical, self.extra_label,
                              self.scope_col, self.data_tag, self.version, self.kind, self.hurdle, self.target_mode,
                              self.target_name])
        return f"T{self.horizon}_{self.scope}_{self.model}{'_' + self.extra_label if self.extra_label else ''}_" \
               + hashlib.md5(payload.encode()).hexdigest()[:8]


def task_data(panel: pd.DataFrame, task: Task, with_target: bool = True) -> pd.DataFrame:
    d = panel[panel[f"has_target_T{task.horizon}"] == 1] if with_target else panel
    if task.scope != "global":
        d = d[d[task.scope_col] == task.scope]
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
    pred = pred[pred["y_true"].notna()]
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


BURN_IN_FOLDS = 1  # dobras iniciais fora do histórico das camadas de magnitude e conformal


def calibration_history(pred: pd.DataFrame, fold: int, burn_in: int = BURN_IN_FOLDS) -> pd.DataFrame:
    """Previsões com alvo das dobras anteriores a `fold`, sem as `burn_in` primeiras (aquecimento).

    A 1ª dobra treina com uma temporada só: o regime de mercado é constante no treino e o nível da
    previsão não é identificável. Seu erro de nível é estrutural (não se repete nas dobras com duas
    ou mais temporadas de treino) e, se entrasse no histórico, deslocaria a recalibração e inflaria o
    *smearing* e o ajuste conformal de todas as dobras seguintes.
    """
    return pred[(pred["fold"] < fold) & (pred["fold"] > burn_in) & pred["y_true"].notna()]


def _recalibration(prev: pd.DataFrame) -> tuple[float, float]:
    """(a, b) da recalibração linear y ≈ a + b·ŷ no histórico (MQO)."""
    b, a = np.polyfit(prev["y_pred"], prev["y_true"], 1)
    return float(a), float(b)


def _edges(values: pd.Series, n: int) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, n + 1)[1:-1]))


def magnitude_layer(pred: pd.DataFrame, group_col: str = "age_band", min_prev: int = 200,
                    burn_in: int = BURN_IN_FOLDS, method: str = "duan", shrink_cells: float = 50.0) -> pd.DataFrame:
    """Converte o log-retorno previsto em magnitudes de ΔV%, sem olhar o futuro.

    Para cada dobra, usa só as previsões fora da amostra das dobras **anteriores** (sem as de
    aquecimento, ver `calibration_history`). A identificação é sequencial: primeiro a recalibração
    linear ŷ' = a + b·ŷ por MQO de y em ŷ (corrige encolhimento ou excesso de dispersão); depois o
    fator de retransformação f = média de exp(y − ŷ') nos resíduos dessa regressão, porque exp(ŷ')
    estima a **mediana** de V_{T+k}/V_T, não a média. Como f é estimado depois de (a, b), exp(a) e f
    não competem pelo mesmo nível.

    `method`:
    * "duan": f por grupo (`group_col`), o *smearing* de Duan (1983) estratificado por idade;
    * "hetero": f por célula decil de ŷ' × tercil de ln V_T × grupo, encolhido para a média do grupo
      com peso `shrink_cells` (Manning e Mullahy, 2001: com variância residual que depende de x, um
      fator constante por grupo subestima a cauda direita).

    Devolve `dv_pct_pred_median` = exp(ŷ') − 1 e `dv_pct_pred_mean` = f·exp(ŷ') − 1. Sem histórico
    suficiente, a camada é a identidade.
    """
    out = []
    for fold in sorted(pred["fold"].unique()):
        cur = pred[pred["fold"] == fold].copy()
        prev = calibration_history(pred, fold, burn_in)
        a, b = 0.0, 1.0
        cur_f = pd.Series(1.0, index=cur.index)
        if len(prev) >= min_prev:
            a, b = _recalibration(prev)
            fitted = a + b * prev["y_pred"]
            ratio = np.exp(prev["y_true"] - fitted)
            band = ratio.groupby(prev[group_col]).mean()
            overall = float(ratio.mean())
            cur_band = cur[group_col].map(band).fillna(overall).astype(float)
            cur_f = cur_band
            if method == "hetero":
                e_pred, e_val = _edges(fitted, 10), _edges(np.log(prev["value_T"]), 3)

                def cells(frame: pd.DataFrame, yhat: pd.Series) -> pd.Series:
                    dec = np.searchsorted(e_pred, yhat.to_numpy(), side="right")
                    ter = np.searchsorted(e_val, np.log(frame["value_T"].to_numpy()), side="right")
                    return pd.Series([f"{d}|{t}|{g}" for d, t, g in zip(dec, ter, frame[group_col].astype(str))],
                                     index=frame.index)

                prev_cells = cells(prev, fitted)
                stats_ = ratio.groupby(prev_cells).agg(["sum", "size"])
                prior = prev_cells.map(lambda c: c.split("|")[2]).map(band.rename(str))
                prior_by_cell = prior.groupby(prev_cells).first()
                f_cell = (stats_["sum"] + shrink_cells * prior_by_cell) / (stats_["size"] + shrink_cells)
                cur_f = cells(cur, a + b * cur["y_pred"]).map(f_cell).fillna(cur_band).astype(float)
        cur["y_pred_cal"] = a + b * cur["y_pred"]
        cur["smearing"] = cur_f
        cur["dv_pct_pred_median"] = np.expm1(cur["y_pred_cal"]) * 100
        cur["dv_pct_pred_mean"] = (np.exp(cur["y_pred_cal"]) * cur["smearing"] - 1) * 100
        out.append(cur)
    return pd.concat(out)


def multiquantile_mean(mq: pd.DataFrame, levels: Sequence[float] | None = None) -> pd.Series:
    """E[ΔV%] pela integração de exp(q_τ) − 1 em τ ∈ (0, 1), regra do ponto médio sobre os níveis do
    CatBoost multiquantil: não paramétrica e heterocedástica por construção, porque a assimetria da
    distribuição prevista de cada atleta entra inteira na média."""
    cols = [mq_column(q) for q in (levels or MQ_LEVELS)]
    return (np.exp(mq[cols]).mean(axis=1) - 1) * 100


def two_part_magnitude(pred: pd.DataFrame, p_col: str = "p_breakout", group_col: str = "age_band",
                       breakout: float = np.log(2.0), min_prev: int = 200, burn_in: int = BURN_IN_FOLDS) -> pd.Series:
    """Modelo em duas partes para a cauda: E[ΔV%] = p·μ_b + (1 − p)·(f_nb·exp(ŷ') − 1).

    p = P(V_{T+k}/V_T ≥ 2 | x) vem de um classificador próprio; μ_b é a valorização média dos saltos
    (y ≥ ln 2) no grupo e f_nb o fator de retransformação dos demais, ambos no histórico das dobras
    anteriores, com a mesma recalibração (a, b) da camada de magnitude.
    """
    out = []
    for fold in sorted(pred["fold"].unique()):
        cur = pred[pred["fold"] == fold]
        prev = calibration_history(pred, fold, burn_in)
        if len(prev) < min_prev:
            out.append(pd.Series(np.expm1(cur["y_pred"]) * 100, index=cur.index))
            continue
        a, b = _recalibration(prev)
        jump = prev["y_true"] >= breakout
        f_nb = np.exp(prev.loc[~jump, "y_true"] - (a + b * prev.loc[~jump, "y_pred"])).groupby(prev.loc[~jump, group_col]).mean()
        mu_b = np.expm1(prev.loc[jump, "y_true"]).groupby(prev.loc[jump, group_col]).mean()
        f = cur[group_col].map(f_nb).fillna(f_nb.mean()).astype(float)
        m = cur[group_col].map(mu_b).fillna(mu_b.mean() if len(mu_b) else 1.0).astype(float)
        base = f * np.exp(a + b * cur["y_pred"]) - 1
        out.append((cur[p_col] * m + (1 - cur[p_col]) * base) * 100)
    return pd.concat(out).reindex(pred.index)


def decile_calibration_error(pred: pd.DataFrame, col: str, n: int = 10) -> pd.DataFrame:
    """Média prevista e observada de ΔV% por decil da previsão, por temporada; o critério de escolha
    da camada de magnitude é o erro absoluto médio entre as duas (inclui o decil superior)."""
    rows = []
    for season, g in pred[pred["y_true"].notna()].groupby("season"):
        dec = pd.qcut(g[col].rank(method="first"), n, labels=False) + 1
        agg = g.groupby(dec).agg(previsto=(col, "mean"), observado=("dv_pct_true", "mean"))
        rows.append(agg.assign(season=season, decil=agg.index))
    out = pd.concat(rows, ignore_index=True)
    out["erro_abs"] = (out["previsto"] - out["observado"]).abs()
    return out


def conformal_interval(pred: pd.DataFrame, lo: str = "q10", hi: str = "q90", nominal: float = 0.8,
                       min_prev: int = 200, burn_in: int = BURN_IN_FOLDS) -> pd.DataFrame:
    """Regressão quantílica conformalizada (Romano et al., 2019), sequencial no tempo.

    Para cada dobra, o escore de conformidade max(q_lo − y, y − q_hi) das dobras **anteriores** (o
    mesmo histórico da camada de magnitude) define o ajuste que leva a cobertura empírica ao nível
    nominal. Sem histórico suficiente, mantém o intervalo bruto.
    """
    out = []
    for fold in sorted(pred["fold"].unique()):
        cur = pred[pred["fold"] == fold].copy()
        prev = calibration_history(pred, fold, burn_in)
        adj = 0.0
        if len(prev) >= min_prev:
            score = np.maximum(prev[lo] - prev["y_true"], prev["y_true"] - prev[hi])
            adj = float(np.quantile(score, min(1.0, nominal * (1 + 1 / len(prev)))))
        cur[f"{lo}_conf"], cur[f"{hi}_conf"], cur["conformal_adj"] = cur[lo] - adj, cur[hi] + adj, adj
        out.append(cur)
    return pd.concat(out)


def interval_metrics(pred: pd.DataFrame, lo: str = "q10", hi: str = "q90", nominal: float = 0.8) -> pd.Series:
    """Cobertura e largura do intervalo preditivo [lo, hi] em log-retorno e na escala de ΔV%."""
    pred = pred[pred["y_true"].notna()]
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
def build_model(spec: ModelSpec, params: dict[str, Any], task: Task) -> Any:
    if spec.key == "median":
        model: Any = MedianBaseline(group_col="category" if task.scope == "global" else None)
    else:
        model = spec.build(params, task.numeric, task.categorical)
        if isinstance(model, EarlyStoppingPipeline):
            model.set_params(purge=task.horizon - 1)
    if task.kind == "reg" and task.target_mode != "nominal":
        model = DriftAdjusted(model, mode=task.target_mode.removeprefix("excess_"), horizon=task.horizon)
    return model


def training_target(task: Task, d: pd.DataFrame) -> np.ndarray:
    """Alvo de treino: o log-retorno ou, na classificação, 1(ΔV% > hurdle)."""
    y = d[task.target].to_numpy(dtype=float)
    return (np.expm1(y) > task.hurdle).astype(int) if task.kind == "clf" else y


def fit_predict(spec: ModelSpec, params: dict[str, Any], task: Task, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    y = training_target(task, train)
    if task.kind == "clf" and len(np.unique(y)) < 2:  # dobra sem uma das classes: a taxa do treino
        return np.full(len(test), float(y.mean()))
    model = build_model(spec, params, task)
    model.fit(train, y)
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

    n_init = max(5, min(10, spec.n_calls // 2))
    res = gp_minimize(objective, spec.space, n_calls=spec.n_calls, n_initial_points=n_init, random_state=seed)
    best = {k: (v.item() if hasattr(v, "item") else v) for k, v in zip(spec.param_names, res.x)}
    return best, [float(v) for v in res.func_vals]


def run_task(task: Task, panel: pd.DataFrame, folds: Sequence[Fold], tuning_folds: Sequence[Fold],
             specs: dict[str, ModelSpec], cache_dir: Path, fixed_params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Otimiza (dobras de validação), avalia em todas as dobras e grava em cache.

    Com `fixed_params`, pula a otimização (usado nos modelos de robustez e de ablação, que
    reaproveitam os hiperparâmetros do modelo principal para isolar o efeito das variáveis). O
    treino e a otimização usam só as linhas com alvo; a previsão cobre todas as linhas de teste do
    escopo, com `has_target` = False para quem ficou sem V_{T+k}.
    """
    path = cache_dir / f"{task.key()}.joblib"
    if path.exists():
        return joblib.load(path)
    t0 = time.time()
    spec = specs[task.model]
    d = task_data(panel, task)
    d_all = task_data(panel, task, with_target=False)
    if fixed_params is not None:
        params, history = dict(fixed_params), []
    else:
        params, history = tune(spec, task, d, tuning_folds)
    preds = []
    for i, fold in enumerate(folds, 1):
        train = d[d["season"].isin(fold.train)]
        test = d_all[d_all["season"] == fold.test]
        if len(train) < 10 or test.empty:
            continue
        p = fit_predict(spec, params, task, train, test)
        h = task.horizon
        frame = pd.DataFrame({
            "season": test["season"].to_numpy(), "fbref_player_id": test["fbref_player_id"].to_numpy(),
            "fbref_team_id": test["fbref_team_id"].to_numpy(), "category": test["category"].to_numpy(),
            "fold": i, "is_holdout": i == len(folds), "has_target": test[f"has_target_T{h}"].eq(1).to_numpy(),
            "y_true": test[task.target].to_numpy(),
            "dv_pct_true": test[f"delta_v_pct_T{h}"].to_numpy(), "dv_true": test[f"delta_v_T{h}"].to_numpy(),
            "value_T": test["value_T"].to_numpy(),
        })
        if task.kind == "mq":  # uma coluna por nível; a mediana é a média dos dois níveis centrais
            q = np.sort(np.asarray(p), axis=1)  # sem cruzamento de quantis
            for j, level in enumerate(MQ_LEVELS):
                frame[mq_column(level)] = q[:, j]
            frame["y_pred"] = q[:, len(MQ_LEVELS) // 2 - 1: len(MQ_LEVELS) // 2 + 1].mean(axis=1)
        else:
            frame["y_pred"] = p
        preds.append(frame)
    result = {"task": task, "params": params, "history": history,
              "predictions": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(),
              "seconds": time.time() - t0, "n_rows": len(d)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, path)
    log.info("tarefa concluída", extra={"key": task.key(), "seconds": round(result["seconds"], 1)})
    return result


def fit_final(spec: ModelSpec, params: dict[str, Any], task: Task, panel: pd.DataFrame,
              seasons: Sequence[str] | None = None) -> Any:
    """Modelo final: hiperparâmetros otimizados, ajustado em todas as temporadas com alvo (ou só em
    `seasons`, para reproduzir o modelo de uma dobra)."""
    d = task_data(panel, task)
    if seasons is not None:
        d = d[d["season"].isin(seasons)]
    return build_model(spec, params, task).fit(d, training_target(task, d))


def mq_column(level: float) -> str:
    return f"q{level:.3f}"


# --------------------------------------------------------------------------------------------
# Pooling parcial entre o modelo segmentado e o global
# --------------------------------------------------------------------------------------------
def pool_predictions(global_pred: pd.Series, segment_pred: pd.Series, n_train: pd.Series, kappa: float) -> pd.Series:
    """Combinação w·segmento + (1 − w)·global, com w = n/(n + κ) e n o treino do segmento na dobra.

    É o encolhimento da previsão de cada segmento para a do modelo global, tanto mais forte quanto
    menor o segmento (κ → ∞ devolve o global; κ = 0, o segmentado). Não exige novos ajustes.
    """
    w = n_train / (n_train + kappa) if np.isfinite(kappa) else 0.0 * n_train
    return w * segment_pred + (1 - w) * global_pred


# --------------------------------------------------------------------------------------------
# Inferência: bootstrap por cluster, teste pareado de perdas, Holm e MCS
# --------------------------------------------------------------------------------------------
CLUSTER_COLS: tuple[str, ...] = ("season", "fbref_team_id")


def cluster_codes(frame: pd.DataFrame, cols: Sequence[str] = CLUSTER_COLS) -> np.ndarray:
    """Código inteiro do cluster de cada linha (por padrão, clube × temporada).

    Atletas do mesmo clube na mesma temporada partilham choques (rebaixamento, troca de técnico,
    revisão do elenco pelo Transfermarkt), por isso a reamostragem e os erros-padrão são por clube.
    """
    return pd.MultiIndex.from_frame(frame[list(cols)].astype(str)).factorize()[0]


def cluster_bootstrap(frame: pd.DataFrame, stat: Callable[[pd.DataFrame], float],
                      clusters: np.ndarray | None = None, n_boot: int = 2000, seed: int = 42,
                      level: float = 0.95) -> dict[str, float]:
    """IC por bootstrap de clusters: sorteia clubes-temporada com reposição e recalcula `stat`."""
    codes = cluster_codes(frame) if clusters is None else np.asarray(clusters)
    members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        draws[b] = stat(frame.iloc[np.concatenate([members[g] for g in pick])])
    alpha = (1 - level) / 2
    return {"estimate": float(stat(frame)), "ci_low": float(np.nanquantile(draws, alpha)),
            "ci_high": float(np.nanquantile(draws, 1 - alpha)), "se": float(np.nanstd(draws, ddof=1)),
            "n_clusters": len(members)}


def _cluster_variance(dev: np.ndarray, codes: np.ndarray, n: int) -> tuple[float, int]:
    sums = pd.Series(dev).groupby(codes).sum()
    g = len(sums)
    return g / (g - 1) * float((sums ** 2).sum()) / n ** 2, g


def paired_loss_test(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, clusters: np.ndarray,
                     loss: str = "squared", clusters2: np.ndarray | None = None) -> dict[str, float]:
    """Teste do tipo Diebold–Mariano para previsões em corte transversal, com erro-padrão por cluster.

    d_i = L(y_i − a_i) − L(y_i − b_i); a média d̄ < 0 indica que `a` erra menos. A variância de d̄
    usa as somas de d por cluster (Cameron, Gelbach e Miller, 2008), com correção G/(G − 1), e o
    p-valor bilateral vem da t de Student com G − 1 graus de liberdade (Harvey, Leybourne e
    Newbold, 1997, para a correção de pequenas amostras).

    Com `clusters2`, a variância é a de dois agrupamentos não aninhados (Cameron, Gelbach e Miller,
    2011): V₁ + V₂ − V₁∩₂, com G = min(G₁, G₂). Nas dobras empilhadas, o mesmo atleta aparece em
    várias temporadas; jogador e clube × temporada cobrem as duas fontes de dependência. Se a soma
    sair negativa (raro), usa a maior das duas variâncias de um agrupamento.
    """
    y, pred_a, pred_b = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    f = np.square if loss == "squared" else np.abs
    d = f(y - pred_a) - f(y - pred_b)
    ok = np.isfinite(d)
    d, codes = d[ok], np.asarray(clusters)[ok]
    n, dbar = len(d), float(d.mean())
    var, g = _cluster_variance(d - dbar, codes, n)
    if clusters2 is not None:
        codes2 = np.asarray(clusters2)[ok]
        inter = pd.MultiIndex.from_arrays([codes, codes2]).factorize()[0]
        var2, g2 = _cluster_variance(d - dbar, codes2, n)
        var12, _ = _cluster_variance(d - dbar, inter, n)
        two_way = var + var2 - var12
        var, g = (two_way if two_way > 0 else max(var, var2)), min(g, g2)
    se = float(np.sqrt(var))
    t = dbar / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=g - 1)) if se > 0 else np.nan
    return {"diff_mean_loss": dbar, "se": se, "t": float(t), "p_value": p, "n": n, "clusters": g}


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """p-valores ajustados por Holm–Bonferroni (controle do erro de família)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, reps: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Conjunto de confiança de modelos (Hansen, Lunde e Nason, 2011), estatística T_R.

    `losses`: uma linha por cluster (perda média dos atletas do clube-temporada) e uma coluna por
    modelo. Os clusters são reamostrados de forma independente (blocos de tamanho 1). Devolve o
    p-valor do MCS de cada modelo e se ele pertence ao conjunto ao nível `size`.
    """
    from arch.bootstrap import MCS

    mcs = MCS(losses, size=size, reps=reps, block_size=1, method="R", seed=seed)
    mcs.compute()
    out = mcs.pvalues.rename(columns={"Pvalue": "p_mcs"})
    out["no_conjunto"] = out.index.isin(mcs.included)
    return out.sort_values("p_mcs", ascending=False)
