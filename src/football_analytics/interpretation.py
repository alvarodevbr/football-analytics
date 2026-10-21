"""Interpretação dos modelos finais (SHAP) e simulação de recrutamento *buy-low / sell-high*.

O modelo final de cada horizonte é a média de três componentes (CatBoost, XGBoost e Ridge) que
compartilham o mesmo espaço de variáveis pré-processadas. Os valores SHAP são aditivos: os do
*ensemble* são a média dos SHAP dos componentes, e o valor-base, a média dos valores-base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .logging_utils import get_logger

log = get_logger("interpretation")


# --------------------------------------------------------------------------------------------
# Carregamento e SHAP
# --------------------------------------------------------------------------------------------
@dataclass
class FinalModel:
    horizon: int
    components: dict[str, object]      # chave do algoritmo -> Pipeline ajustado
    manifest: dict

    @property
    def feature_names(self) -> list[str]:
        return next(iter(self.components.values())).pre_.feature_names_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.components.values()], axis=0)


def load_final(root: Path, horizon: int) -> FinalModel:
    manifest = json.loads((root / "models" / "04_model_training" / "final" / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["horizons"][f"T{horizon}"]
    comps = {key: joblib.load(root / info["path"])["model"] for key, info in entry["global"].items()}
    names = {tuple(m.pre_.feature_names_) for m in comps.values()}
    if len(names) != 1:
        raise ValueError("componentes do ensemble com espaços de variáveis diferentes")
    return FinalModel(horizon, comps, entry)


def component_shap(model: object, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """SHAP de um componente: TreeExplainer nas árvores; forma fechada no Ridge.

    No Ridge, φ_j = w_j (x'_j − E[x'_j]) sobre a variável já winsorizada e escalonada x'. Essa é a
    explicação intervencional exata de um modelo linear, atribuída à variável original.
    """
    import shap

    Z = model.pre_.transform(X)
    est = model.est_
    if hasattr(est, "coef_"):
        mean = Z.mean(axis=0)
        return est.coef_ * (Z - mean), float(est.intercept_ + est.coef_ @ mean)
    explainer = shap.TreeExplainer(est)
    values = explainer.shap_values(Z)
    base = explainer.expected_value
    return np.asarray(values), float(np.ravel(base)[0])


def ensemble_shap(final: FinalModel, X: pd.DataFrame, check: bool = True) -> tuple[pd.DataFrame, float]:
    """SHAP do *ensemble* (média dos componentes), com verificação de aditividade."""
    parts = [component_shap(m, X) for m in final.components.values()]
    phi = np.mean([p[0] for p in parts], axis=0)
    base = float(np.mean([p[1] for p in parts]))
    if check:
        err = np.abs(base + phi.sum(axis=1) - final.predict(X)).max()
        if err > 1e-3:
            raise AssertionError(f"SHAP não aditivo: erro máximo {err:.2e}")
    return pd.DataFrame(phi, columns=final.feature_names, index=X.index), base


def collapse_dummies(phi: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """Soma as colunas `cat=nível` em uma única coluna por variável categórica."""
    out = phi.copy()
    for c in categorical:
        cols = [k for k in out.columns if k.startswith(f"{c}=")]
        if cols:
            out[c] = out[cols].sum(axis=1)
            out = out.drop(columns=cols)
    return out


def importance(phi: pd.DataFrame, by: pd.Series | None = None) -> pd.DataFrame:
    """Média de |SHAP| por variável (globalmente ou por grupo de linhas)."""
    a = phi.abs()
    return a.mean().to_frame("mean_abs_shap") if by is None else a.groupby(by.to_numpy()).mean().T


# --------------------------------------------------------------------------------------------
# Simulação de carteiras
# --------------------------------------------------------------------------------------------
def portfolio_metrics(sel: pd.DataFrame) -> pd.Series:
    """Resultado realizado de uma carteira de atletas comprados por V_T e avaliados em V_{T+k}."""
    pct = sel["dv_pct_true"]
    roi = sel["dv_true"].sum() / sel["value_T"].sum() * 100
    return pd.Series({
        "n": len(sel), "investimento_mi": sel["value_T"].sum() / 1e6,
        "retorno_medio_pct": pct.mean(), "retorno_mediano_pct": pct.median(),
        "roi_carteira_pct": roi, "ganho_liquido_mi": sel["dv_true"].sum() / 1e6,
        "taxa_valorizacao": (sel["y_true"] > 0).mean(), "taxa_perda_grave": (pct < -50).mean(),
    })


def top_k(cands: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    return cands.nlargest(k, score)


def budget_greedy(cands: pd.DataFrame, score: str, budget: float, max_price: float | None = None,
                  max_players: int | None = None) -> pd.DataFrame:
    """Compra os atletas em ordem decrescente de `score` até esgotar o orçamento (preço = V_T)."""
    pool = cands if max_price is None else cands[cands["value_T"] <= max_price]
    chosen, spent = [], 0.0
    for idx, price in pool.sort_values(score, ascending=False)["value_T"].items():
        if spent + price <= budget:
            chosen.append(idx)
            spent += price
            if max_players and len(chosen) >= max_players:
                break
    return cands.loc[chosen]


def random_portfolios(cands: pd.DataFrame, k: int, n_draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Distribuição de referência: carteiras de k atletas sorteados ao acaso."""
    idx = np.array([rng.choice(len(cands), size=k, replace=False) for _ in range(n_draws)])
    pct = cands["dv_pct_true"].to_numpy()[idx]
    dv, v = cands["dv_true"].to_numpy()[idx], cands["value_T"].to_numpy()[idx]
    return pd.DataFrame({"retorno_medio_pct": pct.mean(axis=1), "roi_carteira_pct": dv.sum(axis=1) / v.sum(axis=1) * 100,
                         "taxa_valorizacao": (pct > 0).mean(axis=1)})


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Value-at-Risk e Conditional VaR no nível `alpha` (retornos em %, perda = cauda esquerda)."""
    q = float(np.quantile(returns, alpha))
    return q, float(returns[returns <= q].mean())
