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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .logging_utils import get_logger
from .modeling import BURN_IN_FOLDS, DriftAdjusted

log = get_logger("interpretation")


# --------------------------------------------------------------------------------------------
# Carregamento e SHAP
# --------------------------------------------------------------------------------------------
def pipeline_of(model: object) -> object:
    """O pipeline (pré-processador + estimador) de um componente, sem o invólucro do alvo em excesso."""
    return model.inner_ if isinstance(model, DriftAdjusted) else model


def variable_of(column: str) -> str:
    """Variável original de uma coluna pré-processada: `x×célula` (interação do Ridge) → `x`."""
    return column.split("×")[0]


@dataclass
class FinalModel:
    horizon: int
    components: dict[str, object]      # chave do algoritmo -> Pipeline ajustado (ou DriftAdjusted)
    manifest: dict

    @property
    def feature_names(self) -> list[str]:
        """União das colunas dos componentes, com as interações do Ridge somadas à variável original."""
        names: list[str] = []
        for m in self.components.values():
            names += [v for v in dict.fromkeys(map(variable_of, pipeline_of(m).pre_.feature_names_)) if v not in names]
        return names

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.components.values()], axis=0)


def fold_model(horizon: int, components: dict[str, object]) -> FinalModel:
    """Ensemble de uma dobra (componentes já ajustados no treino da dobra), sem manifesto."""
    return FinalModel(horizon, components, {})


def load_final(root: Path, horizon: int, final_dir: Path | None = None) -> FinalModel:
    final_dir = final_dir or root / "models" / "04_model_training" / "final"
    manifest = json.loads((final_dir / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["horizons"][f"T{horizon}"]
    comps = {key: joblib.load(root / info["path"])["model"] for key, info in entry["global"].items()}
    return FinalModel(horizon, comps, entry)


def component_shap(model: object, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """SHAP de um componente: TreeExplainer nas árvores; forma fechada no Ridge.

    No Ridge, φ_j = w_j (x'_j − E[x'_j]) sobre a variável já winsorizada e escalonada x'. Essa é a
    explicação intervencional exata de um modelo linear, atribuída à variável original; os termos de
    interação `x × célula` somam-se a x. Com o alvo em excesso ao *drift* de mercado, o *drift* previsto
    é aditivo: a média entra no valor-base e o desvio (só no modo "regime", que depende do regime de
    mercado de T) é atribuído à variável de regime.
    """
    import shap

    pipe = pipeline_of(model)
    Z = pipe.pre_.transform(X)
    est = pipe.est_
    if hasattr(est, "coef_"):
        mean = Z.mean(axis=0)
        values, base = est.coef_ * (Z - mean), float(est.intercept_ + est.coef_ @ mean)
    else:
        explainer = shap.TreeExplainer(est)
        values, base = np.asarray(explainer.shap_values(Z)), float(np.ravel(explainer.expected_value)[0])
    phi = pd.DataFrame(values, columns=pipe.pre_.feature_names_, index=X.index)
    phi = phi.T.groupby(phi.columns.map(variable_of), sort=False).sum().T
    if isinstance(model, DriftAdjusted):
        drift = model.drift_forecast(X)
        base += float(drift.mean())
        if model.mode == "regime" and model.regime_col in phi:
            phi[model.regime_col] += drift - drift.mean()
    return phi, base


def ensemble_shap(final: FinalModel, X: pd.DataFrame, check: bool = True) -> tuple[pd.DataFrame, float]:
    """SHAP do *ensemble* (média dos componentes), com verificação de aditividade."""
    names = final.feature_names
    parts = [component_shap(m, X) for m in final.components.values()]
    phi = sum(p[0].reindex(columns=names, fill_value=0.0) for p in parts) / len(parts)
    base = float(np.mean([p[1] for p in parts]))
    if check:
        err = np.abs(base + phi.sum(axis=1).to_numpy() - final.predict(X)).max()
        if err > 1e-3:
            raise AssertionError(f"SHAP não aditivo: erro máximo {err:.2e}")
    return phi, base


def coalition_shap(phi: pd.DataFrame, coalitions: dict[str, Sequence[str]]) -> pd.DataFrame:
    """SHAP de coalizões: pela aditividade, o efeito de um grupo de variáveis é a soma exata dos seus
    valores SHAP. Junta variáveis que são funções umas das outras (idade, idade² e faixa etária; nível de
    preço e gaps de avaliação), que as árvores usam de forma intercambiável e a figura fragmentaria."""
    out = phi.copy()
    for name, cols in coalitions.items():
        present = [c for c in cols if c in out]
        if present:
            out[name] = out[present].sum(axis=1)
            out = out.drop(columns=present)
    return out


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
MISSING_OUTCOMES = ("exclude", "zero", "last", "reason")


def outcome_value(sel: pd.DataFrame, missing: str = "exclude") -> pd.Series:
    """Valor em T+k de cada atleta da carteira, com um tratamento explícito para quem saiu da base.

    * ``exclude``: atletas sem V_{T+k} saem da carteira (o *backtest* condicionado à sobrevivência);
    * ``zero``: o ativo vale zero (aposentadoria; limite pessimista);
    * ``last``: o último valor publicado até T+k, sem limite de defasagem (limite otimista; coluna
      ``value_last_obs``);
    * ``reason`` (cenário principal da v4): zero só para quem se aposentou, ficou sem clube ou parou a
      carreira (``target_missing_reason`` = ``career_end``); quem só ficou sem avaliação recente, em
      geral numa liga menor, vale o último valor publicado.
    """
    if missing not in MISSING_OUTCOMES:
        raise ValueError(f"tratamento desconhecido: {missing}")
    observed = sel["value_T"] + sel["dv_true"]
    if missing == "exclude":
        return observed.where(sel["dv_true"].notna())
    last = sel["value_last_obs"].fillna(sel["value_T"])
    if missing == "reason":
        fill = last.where(sel["target_missing_reason"] != "career_end", 0.0)
    else:
        fill = 0.0 if missing == "zero" else last
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
                  max_players: int | None = None, max_per_group: dict[str, int] | None = None,
                  buy_cost: float = 0.0) -> pd.DataFrame:
    """Compra os atletas em ordem decrescente de `score` até esgotar o orçamento (preço = V_T(1 + buy_cost)).

    `max_per_group` limita a concentração da carteira, por exemplo ``{"position_group": 2,
    "age_band": 3}``: no máximo dois atletas por posição e três por faixa etária. Com `buy_cost` = 0,
    reproduz a regra da v3, que conferia o orçamento sem o custo de compra.
    """
    pool = cands if max_price is None else cands[cands["value_T"] <= max_price]
    caps = max_per_group or {}
    floor = float(pool["value_T"].min()) * (1 + buy_cost) if len(pool) else 0.0
    chosen, spent = [], 0.0
    counts: dict[tuple[str, object], int] = {}
    ordered = pool.sort_values(score, ascending=False)[["value_T", *caps]]
    ordered = ordered.assign(value_T=ordered["value_T"] * (1 + buy_cost))
    for idx, price, *levels in ordered.itertuples(name=None):
        if spent + floor > budget:   # nenhum candidato restante cabe no orçamento
            break
        if spent + price > budget:
            continue
        keys = list(zip(caps, levels))
        if any(counts.get(key, 0) >= caps[key[0]] for key in keys):
            continue
        chosen.append(idx)
        spent += price
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        if max_players and len(chosen) >= max_players:
            break
    return cands.loc[chosen]


def budget_milp(cands: pd.DataFrame, score: str, budget: float, buy_cost: float = 0.0,
                max_price: float | None = None, max_players: int | None = None,
                max_per_group: dict[str, int] | None = None, downside: str | None = None,
                hurdle: float = 0.0, lam: float = 0.0) -> pd.DataFrame:
    """Carteira ótima sob orçamento por programação inteira (mochila com restrições de composição).

    Maximiza Σ z_i · [score_i − λ · max(0, hurdle − downside_i)], com z_i ∈ {0, 1}, sujeito a
    Σ z_i · V_i(1 + buy_cost) ≤ budget (o preço já inclui o custo de compra), ao número máximo de
    atletas e aos limites por grupo. `score` e `downside` estão em ΔV% (pontos percentuais) e `hurdle`
    em fração: o termo penaliza quem tem P10 conformalizado abaixo do ponto de equilíbrio, unindo a
    previsão pontual e a incerteza num só critério. Só entram candidatos com utilidade positiva.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    pool = cands if max_price is None else cands[cands["value_T"] <= max_price]
    util = pool[score].astype(float)
    if downside is not None and lam > 0:
        util = util - lam * (hurdle * 100 - pool[downside].astype(float)).clip(lower=0)
    pool, util = pool[util > 0], util[util > 0]
    if pool.empty:
        return cands.iloc[:0]
    rows, lo, hi = [(pool["value_T"] * (1 + buy_cost)).to_numpy()], [0.0], [budget]
    if max_players:
        rows.append(np.ones(len(pool)))
        lo.append(0.0)
        hi.append(max_players)
    for col, cap in (max_per_group or {}).items():
        for level in pool[col].dropna().unique():
            rows.append((pool[col] == level).to_numpy(dtype=float))
            lo.append(0.0)
            hi.append(cap)
    res = milp(-util.to_numpy(), constraints=LinearConstraint(np.vstack(rows), lo, hi),
               integrality=np.ones(len(pool)), bounds=Bounds(0, 1))
    if res.x is None:
        return cands.iloc[:0]
    return pool[res.x > 0.5]


def buyable(cands: pd.DataFrame, top_clubs: int = 10, minutes_share_max: float = 15.0,
            club_col: str = "fbref_team_id", rank_col: str = "club_value_rank_pct") -> pd.Series:
    """Universo comprável: exclui os atletas dos `top_clubs` elencos mais valiosos das Big Five na
    temporada, salvo quem joga menos de `minutes_share_max`% dos minutos (sinal de empréstimo ou venda)."""
    clubs = cands[["season", club_col, rank_col]].drop_duplicates(["season", club_col])
    elite = clubs.groupby("season", group_keys=False).apply(lambda g: g.nlargest(top_clubs, rank_col))
    is_elite = pd.MultiIndex.from_frame(cands[["season", club_col]]).isin(
        pd.MultiIndex.from_frame(elite[["season", club_col]]))
    return pd.Series(~is_elite | (cands["minutes_share_pct"] < minutes_share_max).to_numpy(), index=cands.index)


def acquisition_premium(rows: pd.DataFrame, transfers: pd.DataFrame, as_of: pd.Timestamp,
                        by: Sequence[str] = ("age_band", "elite_seller"), min_n: int = 20) -> pd.DataFrame:
    """Ágio de aquisição observado: mediana de taxa de transferência / V_T nas vendas definitivas com
    taxa paga nos 12 meses após 30/jun de T, saindo do clube do atleta em T.

    Só entram transferências até `as_of` (a data de compra simulada), para que o cenário de estresse
    não use informação futura. `rows` traz `tm_player_id`, `season`, `value_T` e as colunas de `by`;
    grupos com menos de `min_n` vendas recebem a mediana geral.
    """
    from .config import Season

    t = transfers[(transfers["fee_eur"] > 0) & ~transfers["is_loan"].fillna(False).astype(bool)].copy()
    t["transfer_date"] = pd.to_datetime(t["transfer_date"])
    t = t[t["transfer_date"] <= as_of]
    r = rows.assign(anchor=rows["season"].map(lambda s: pd.Timestamp(Season(s).end_anchor)))
    m = r.merge(t[["tm_player_id", "transfer_date", "fee_eur"]], on="tm_player_id")
    m = m[(m["transfer_date"] > m["anchor"]) & (m["transfer_date"] <= m["anchor"] + pd.Timedelta(days=365))]
    m = m.sort_values("transfer_date").drop_duplicates(["tm_player_id", "season"])
    m["ratio"] = m["fee_eur"] / m["value_T"]
    overall = float(m["ratio"].median()) if len(m) else np.nan
    g = m.groupby(list(by))["ratio"].agg(["median", "size"]).reset_index()
    g["premio"] = np.where(g["size"] >= min_n, g["median"], overall)
    return g.assign(mediana_geral=overall, n_total=len(m))


def wilson_interval(hits: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """IC de Wilson para uma proporção (acerto de uma carteira de n atletas)."""
    from scipy.stats import norm

    if n == 0:
        return np.nan, np.nan
    z = norm.ppf(0.5 + level / 2)
    p = hits / n
    centre = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / (1 + z ** 2 / n)
    return float(centre - half), float(centre + half)


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


# --------------------------------------------------------------------------------------------
# Avaliação de decisão: a valorização paga a operação?
# --------------------------------------------------------------------------------------------
def break_even_return(buy_cost: float = 0.0, sell_cost: float = 0.0, horizon: int = 1, rate: float = 0.0,
                      holding: float = 0.0) -> float:
    """ΔV% (fração de V_T) acima do qual comprar por V_T(1 + b) e vender por V_{T+k}(1 − s) dá lucro.

    Com `rate` (custo de oportunidade anual) e `holding` (custo de carregamento anual, em fração de
    V_T: salário líquido do retorno esportivo), o ponto de equilíbrio cresce com o horizonte k:
    [(1 + b)(1 + r)^k + k·c_h] / (1 − s) − 1. Sem eles, é o mesmo em T+1 e T+2.
    """
    return ((1 + buy_cost) * (1 + rate) ** horizon + horizon * holding) / (1 - sell_cost) - 1


def pays_costs(dv_pct: pd.Series, buy_cost: float = 0.0, sell_cost: float = 0.0) -> pd.Series:
    """1 se a valorização realizada (ΔV%, em %) supera o ponto de equilíbrio dos custos, senão 0.

    Atletas sem V_{T+k} contam como prejuízo total, como no cenário principal do *backtest* (quem sai
    da base vale zero). A tolerância evita que um ΔV% exatamente no equilíbrio conte como lucro.
    """
    net = (1 + dv_pct.fillna(-100.0) / 100) * (1 - sell_cost) / (1 + buy_cost) - 1
    return (net > 1e-9).astype(int)


def auc(label: np.ndarray | pd.Series, score: np.ndarray | pd.Series) -> float:
    """Área sob a curva ROC pela estatística de Mann–Whitney (empates valem meio); ignora escores ausentes."""
    from scipy.stats import rankdata

    y, s = np.asarray(label, dtype=float), np.asarray(score, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def decision_metrics(label: pd.Series, score: pd.Series, top: float = 0.10) -> dict[str, float]:
    """Qualidade de um escore como regra de compra.

    Discriminação (AUC), precisão média (AP) e acerto nos extremos: a fração dos `top` maiores e dos
    `top` menores escores cujo rótulo é 1. O tamanho dos extremos segue o *backtest*
    (``len // 10`` com ``top = 0,10``); candidatos sem escore ficam de fora.
    """
    from sklearn.metrics import average_precision_score

    ok = score.notna()
    y, s = label[ok].to_numpy(dtype=float), score[ok].to_numpy(dtype=float)
    n = max(1, int(len(s) * top))
    order = np.argsort(-s, kind="stable")
    return {"n": len(s), "taxa_base": float(y.mean()), "auc": auc(y, s),
            "ap": float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else np.nan,
            "acerto_topo": float(y[order[:n]].mean()), "acerto_base": float(y[order[-n:]].mean())}


def exceedance_probability(pred: pd.DataFrame, score: str, label: str, min_prev: int = 200,
                           burn_in: int = BURN_IN_FOLDS) -> pd.DataFrame:
    """Probabilidade fora da amostra de `label` = 1 dado `score`, por regressão isotônica.

    `pred` é de um só horizonte. A dobra f é calibrada só com as dobras anteriores, sem as `burn_in`
    primeiras: a mesma regra de histórico da camada de magnitude e dos intervalos conformalizados. Com
    menos de `min_prev` observações no histórico, a dobra fica sem probabilidade. Retorna ``prob`` e
    ``clim``, a taxa-base do mesmo histórico (a referência do escore de Brier).
    """
    from sklearn.isotonic import IsotonicRegression

    out = pd.DataFrame({"prob": np.nan, "clim": np.nan}, index=pred.index)
    for fold in sorted(pred["fold"].unique()):
        hist = pred[(pred["fold"] < fold) & (pred["fold"] > burn_in) & pred[score].notna()]
        if len(hist) < min_prev:
            continue
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(hist[score], hist[label])
        cur = (pred["fold"] == fold) & pred[score].notna()
        out.loc[cur, "prob"] = iso.predict(pred.loc[cur, score])
        out.loc[cur, "clim"] = hist[label].mean()
    return out


def reliability_table(prob: pd.Series, label: pd.Series,
                      bins: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0)) -> pd.DataFrame:
    """Diagrama de confiabilidade: probabilidade média prevista × frequência observada por faixa."""
    ok = prob.notna()
    frame = pd.DataFrame({"p": prob[ok], "y": label[ok],
                          "faixa": pd.cut(prob[ok], list(bins), include_lowest=True)})
    g = frame.groupby("faixa", observed=True)
    table = pd.DataFrame({"n": g.size(), "prob_prevista": g["p"].mean(), "freq_observada": g["y"].mean()})
    return table.reset_index().assign(faixa=lambda t: t["faixa"].astype(str))
