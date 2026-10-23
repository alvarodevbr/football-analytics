"""Interpretação dos modelos finais (SHAP) e simulação de recrutamento *buy-low / sell-high*.

O modelo final de cada horizonte é a média de três componentes (CatBoost, XGBoost e Ridge) que
compartilham o mesmo espaço de variáveis pré-processadas. Os valores SHAP são aditivos: os do
*ensemble* são a média dos SHAP dos componentes, e o valor-base, a média dos valores-base.

Além do modelo final, os modelos de cada dobra da janela expandida são reajustados para explicar as
previsões fora da amostra que eles próprios geraram (estabilidade das explicações e *waterfalls*
coerentes com a previsão exibida).
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


def fold_model(horizon: int, components: dict[str, object]) -> FinalModel:
    """Ensemble de uma dobra (componentes já ajustados no treino da dobra), sem manifesto."""
    names = {tuple(m.pre_.feature_names_) for m in components.values()}
    if len(names) != 1:
        raise ValueError("componentes do ensemble com espaços de variáveis diferentes")
    return FinalModel(horizon, components, {})


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


def group_shares(phi: pd.DataFrame, group_of: dict[str, str]) -> pd.Series:
    """Participação (%) de cada grupo de variáveis no |SHAP| total."""
    total = phi.abs().T.groupby([group_of[c] for c in phi.columns]).sum().T.sum()
    return total / total.sum() * 100


def importance_stability(importances: dict[object, pd.Series], top: int = 20) -> pd.DataFrame:
    """τ de Kendall entre os rankings de importância de cada par de dobras.

    Usa as `top` variáveis mais importantes na média das dobras; τ próximo de 1 indica que o modelo
    de cada temporada se apoia nas mesmas variáveis, na mesma ordem.
    """
    from scipy.stats import kendalltau

    table = pd.DataFrame(importances)
    keep = table.mean(axis=1).sort_values(ascending=False).head(top).index
    keys = list(table.columns)
    rows = [{"dobra_a": a, "dobra_b": b, "tau_kendall": float(kendalltau(table.loc[keep, a], table.loc[keep, b]).statistic)}
            for i, a in enumerate(keys) for b in keys[i + 1:]]
    return pd.DataFrame(rows)


def group_share_bootstrap(phi: pd.DataFrame, group_of: dict[str, str], by: pd.Series, clusters: np.ndarray,
                          n_boot: int = 1000, seed: int = 42) -> pd.DataFrame:
    """IC de 95% da participação de cada grupo de variáveis no |SHAP|, por nível de `by` (ex.: faixa
    etária), com reamostragem de clusters (clube × temporada) dentro de cada nível."""
    groups = sorted(set(group_of.values()))
    absg = phi.abs().T.groupby([group_of[c] for c in phi.columns]).sum().T.reindex(columns=groups, fill_value=0)
    rng = np.random.default_rng(seed)
    rows = []
    for level in pd.unique(by):
        mask = (by == level).to_numpy()
        sub, codes = absg[mask].to_numpy(), np.asarray(clusters)[mask]
        uniq = np.unique(codes)
        members = [np.flatnonzero(codes == u) for u in uniq]
        point = sub.sum(axis=0) / sub.sum() * 100
        draws = np.empty((n_boot, len(groups)))
        for b in range(n_boot):
            idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
            tot = sub[idx].sum(axis=0)
            draws[b] = tot / tot.sum() * 100
        for j, g in enumerate(groups):
            rows.append({"nivel": level, "grupo": g, "participacao_pct": point[j],
                         "ic95_inf": np.quantile(draws[:, j], 0.025), "ic95_sup": np.quantile(draws[:, j], 0.975)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Simulação de carteiras
# --------------------------------------------------------------------------------------------
MISSING_OUTCOMES = ("exclude", "zero", "last")


def outcome_value(sel: pd.DataFrame, missing: str = "exclude") -> pd.Series:
    """Valor em T+k de cada atleta da carteira, com um tratamento explícito para quem saiu da base.

    * ``exclude``: atletas sem V_{T+k} saem da carteira (o *backtest* condicionado à sobrevivência);
    * ``zero``: o ativo vale zero (aposentadoria; limite pessimista);
    * ``last``: o último valor publicado até T+k, sem limite de defasagem (limite otimista; coluna
      ``value_last_obs``).
    """
    if missing not in MISSING_OUTCOMES:
        raise ValueError(f"tratamento desconhecido: {missing}")
    observed = sel["value_T"] + sel["dv_true"]
    if missing == "exclude":
        return observed.where(sel["dv_true"].notna())
    fill = 0.0 if missing == "zero" else sel["value_last_obs"].fillna(sel["value_T"])
    return observed.fillna(fill)


def portfolio_metrics(sel: pd.DataFrame, buy_cost: float = 0.0, sell_cost: float = 0.0,
                      missing: str = "exclude") -> pd.Series:
    """Resultado realizado de uma carteira comprada por V_T(1 + `buy_cost`) e vendida por
    V_{T+k}(1 − `sell_cost`), com o tratamento `missing` para quem ficou sem V_{T+k}.

    Sem custos e com ``missing="exclude"``, reproduz o ROI de valor de mercado ΣΔV / ΣV_T.
    """
    v1 = outcome_value(sel, missing)
    ok = v1.notna()
    cost, proceeds = sel.loc[ok, "value_T"] * (1 + buy_cost), v1[ok] * (1 - sell_cost)
    pct = (proceeds / cost - 1) * 100
    gain = proceeds.sum() - cost.sum()
    return pd.Series({
        "n": int(ok.sum()), "sem_alvo": int((~sel["dv_true"].notna()).sum()),
        "investimento_mi": cost.sum() / 1e6,
        "retorno_medio_pct": pct.mean(), "retorno_mediano_pct": pct.median(),
        "roi_carteira_pct": gain / cost.sum() * 100 if cost.sum() > 0 else np.nan, "ganho_liquido_mi": gain / 1e6,
        "taxa_valorizacao": (pct > 0).mean(), "taxa_perda_grave": (pct < -50).mean(),
    })


def break_even_buy_cost(sel: pd.DataFrame, sell_cost: float = 0.0, missing: str = "exclude") -> float:
    """Ágio de compra (fração de V_T) que zera o ROI da carteira: ΣV_{T+k}(1 − s)/ΣV_T − 1."""
    v1 = outcome_value(sel, missing)
    ok = v1.notna()
    return float((v1[ok] * (1 - sell_cost)).sum() / sel.loc[ok, "value_T"].sum() - 1)


def roi_without_top(sel: pd.DataFrame, k: int = 1, missing: str = "exclude") -> float:
    """ROI (%) da carteira sem os `k` atletas de maior ganho absoluto: robustez a poucos acertos."""
    v1 = outcome_value(sel, missing)
    gain = (v1 - sel["value_T"]).dropna()
    keep = gain.drop(gain.nlargest(k).index).index
    return float(gain[keep].sum() / sel.loc[keep, "value_T"].sum() * 100)


def sign_test(wins: int, n: int) -> float:
    """p-valor unilateral de vencer `wins` de `n` temporadas sob H0 de 50% (teste do sinal)."""
    from scipy.stats import binom

    return float(binom.sf(wins - 1, n, 0.5))


def top_k(cands: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    return cands.nlargest(k, score)


def budget_greedy(cands: pd.DataFrame, score: str, budget: float, max_price: float | None = None,
                  max_players: int | None = None, max_per_group: dict[str, int] | None = None) -> pd.DataFrame:
    """Compra os atletas em ordem decrescente de `score` até esgotar o orçamento (preço = V_T).

    `max_per_group` limita a concentração da carteira, por exemplo ``{"position_group": 2,
    "age_band": 3}``: no máximo dois atletas por posição e três por faixa etária.
    """
    pool = cands if max_price is None else cands[cands["value_T"] <= max_price]
    chosen, spent = [], 0.0
    counts: dict[tuple[str, object], int] = {}
    for idx, row in pool.sort_values(score, ascending=False).iterrows():
        price = row["value_T"]
        if spent + price > budget:
            continue
        if max_per_group and any(counts.get((col, row[col]), 0) >= cap for col, cap in max_per_group.items()):
            continue
        chosen.append(idx)
        spent += price
        for col in (max_per_group or {}):
            counts[(col, row[col])] = counts.get((col, row[col]), 0) + 1
        if max_players and len(chosen) >= max_players:
            break
    return cands.loc[chosen]


def random_portfolios(cands: pd.DataFrame, k: int, n_draws: int, rng: np.random.Generator,
                      buy_cost: float = 0.0, sell_cost: float = 0.0, missing: str = "exclude") -> pd.DataFrame:
    """Distribuição de referência: carteiras de k atletas sorteados ao acaso, com os mesmos custos e o
    mesmo tratamento dos atletas sem V_{T+k} da carteira do modelo (ver `portfolio_metrics`)."""
    v1_all = outcome_value(cands, missing)
    pool = cands[v1_all.notna()]
    v1 = v1_all[v1_all.notna()].to_numpy()
    k = min(k, len(pool))
    idx = np.array([rng.choice(len(pool), size=k, replace=False) for _ in range(n_draws)])
    cost = pool["value_T"].to_numpy()[idx] * (1 + buy_cost)
    proceeds = v1[idx] * (1 - sell_cost)
    pct = (proceeds / cost - 1) * 100
    return pd.DataFrame({"retorno_medio_pct": pct.mean(axis=1),
                         "roi_carteira_pct": (proceeds.sum(axis=1) / cost.sum(axis=1) - 1) * 100,
                         "taxa_valorizacao": (pct > 0).mean(axis=1)})


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Value-at-Risk e Conditional VaR no nível `alpha` (retornos em %, perda = cauda esquerda)."""
    q = float(np.quantile(returns, alpha))
    return q, float(returns[returns <= q].mean())
