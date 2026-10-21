"""Construção da amostra analítica: unidade jogador–temporada, posição, idade, 12 categorias,
alvos de variação de valor e o funil de filtros (goleiros, minutagem, p90).

Usado pelo notebook 02 (validação dos cortes) e pelo 03 (aplicação), para que os dois partam
exatamente das mesmas regras.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import Season

KEY: list[str] = ["season", "fbref_player_id"]

POSITIONS: tuple[str, ...] = ("DEF", "MEI", "ATA")
POSITION_LABELS: dict[str, str] = {"DEF": "Defensor", "MEI": "Meio-campista", "ATA": "Atacante"}

# Posição granular do Transfermarkt -> posição de linha do estudo. O meia ofensivo fica no meio-campo,
# como o próprio Transfermarkt o classifica; pontas e segundo atacante vão para o ataque.
TM_POSITION_GROUP: dict[str, str] = {
    "Goalkeeper": "GK",
    "Centre-Back": "DEF", "Left-Back": "DEF", "Right-Back": "DEF", "Defender": "DEF",
    "Defensive Midfield": "MEI", "Central Midfield": "MEI", "Attacking Midfield": "MEI",
    "Left Midfield": "MEI", "Right Midfield": "MEI", "Midfield": "MEI",
    "Left Winger": "ATA", "Right Winger": "ATA", "Centre-Forward": "ATA", "Second Striker": "ATA",
    "Attack": "ATA",
}
FBREF_POSITION_GROUP: dict[str, str] = {"GK": "GK", "DF": "DEF", "MF": "MEI", "FW": "ATA"}

# Faixas etárias em anos completos na data-âncora de T. Leitura formal dos intervalos da
# especificação (Tabela 6.2): [16, 23], (23, 28], (28, 32], (32, ∞).
AGE_BANDS: tuple[tuple[str, int, int], ...] = (
    ("16–23", 0, 23),
    ("24–28", 24, 28),
    ("29–32", 29, 32),
    ("33+", 33, 200),
)
AGE_BAND_LABELS: tuple[str, ...] = tuple(b[0] for b in AGE_BANDS)

LEAGUE_LABELS: dict[str, str] = {
    "ENG-Premier League": "Premier League",
    "ESP-La Liga": "La Liga",
    "ITA-Serie A": "Serie A",
    "GER-Bundesliga": "Bundesliga",
    "FRA-Ligue 1": "Ligue 1",
}


def season_label(code: str) -> str:
    """``"2324"`` -> ``"2023/24"``."""
    return f"20{code[:2]}/{code[2:]}"


def position_group(tm_position: pd.Series, fb_position: pd.Series) -> pd.Series:
    """Posição de linha: Transfermarkt (granular) com fallback na primeira posição do FBref."""
    from_tm = tm_position.map(TM_POSITION_GROUP)
    from_fb = fb_position.fillna("").str.split(",").str[0].str.strip().map(FBREF_POSITION_GROUP)
    return from_tm.fillna(from_fb)


def completed_years(birth: pd.Series, reference: pd.Series) -> pd.Series:
    """Idade em anos completos de `birth` até `reference` (ambas datetime64)."""
    years = reference.dt.year - birth.dt.year
    before_birthday = (reference.dt.month < birth.dt.month) | (
        (reference.dt.month == birth.dt.month) & (reference.dt.day < birth.dt.day)
    )
    return (years - before_birthday.astype(int)).where(birth.notna() & reference.notna()).astype("Int64")


def age_band(age: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=age.index, dtype="object")
    for label, low, high in AGE_BANDS:
        out[(age >= low) & (age <= high)] = label
    return pd.Categorical(out, categories=AGE_BAND_LABELS, ordered=True)


def category_id(band: pd.Series, position: pd.Series) -> pd.Series:
    """C1–C12 na ordem da Tabela 5.2: faixa etária (linhas) × DEF/MEI/ATA (colunas)."""
    band_idx = pd.Series(pd.Categorical(band, categories=AGE_BAND_LABELS).codes, index=band.index)
    pos_idx = position.map({p: i for i, p in enumerate(POSITIONS)})
    valid = (band_idx >= 0) & pos_idx.notna()
    number = band_idx * len(POSITIONS) + pos_idx.fillna(0).astype(int) + 1
    return ("C" + number.astype(str)).where(valid)


def category_labels() -> dict[str, str]:
    """``{"C1": "C1 · DEF 16–23", ...}``."""
    labels = {}
    for b, band in enumerate(AGE_BAND_LABELS):
        for p, pos in enumerate(POSITIONS):
            cid = f"C{b * len(POSITIONS) + p + 1}"
            labels[cid] = f"{cid} · {pos} {band}"
    return labels


def to_player_season(rows: pd.DataFrame, sum_cols: tuple[str, ...] = ("fb_minutes",)) -> pd.DataFrame:
    """Colapsa jogador–clube–temporada em jogador–temporada.

    Atletas transferidos dentro da temporada têm uma linha por clube no FBref, mas um único valor de
    mercado. Os minutos (e outras contagens em `sum_cols`) são somados; os demais atributos vêm do
    clube em que o atleta jogou mais minutos.
    """
    ordered = rows.sort_values(["fb_minutes"], ascending=False, na_position="last")
    main = ordered.drop_duplicates(KEY).drop(columns=list(sum_cols))
    sums = rows.groupby(KEY, as_index=False)[list(sum_cols)].sum(min_count=1)
    counts = rows.groupby(KEY).size().rename("n_club_rows").reset_index()
    return main.merge(sums, on=KEY).merge(counts, on=KEY)


def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Posição de linha, idade na âncora de T, faixa etária e categoria C1–C12."""
    out = df.copy()
    out["position_group"] = position_group(out["tm_position"], out["fb_position"])
    anchors = {code: pd.Timestamp(Season(code).end_anchor) for code in out["season"].unique()}
    out["anchor_date_T"] = out["season"].map(anchors)
    birth = pd.to_datetime(out["tm_birth_date"], errors="coerce")
    # Sem data de nascimento do TM, usa o ano do FBref com aniversário assumido em 1º de julho.
    fallback = pd.to_datetime(out["fb_birth_year"].astype("Int64").astype(str) + "-07-01", errors="coerce")
    out["birth_date_is_estimated"] = birth.isna() & fallback.notna()
    out["birth_date"] = birth.fillna(fallback)
    out["age_T"] = completed_years(out["birth_date"], out["anchor_date_T"])
    out["age_band"] = age_band(out["age_T"])
    out["category"] = category_id(pd.Series(out["age_band"], index=out.index), out["position_group"])
    return out


def add_value_targets(df: pd.DataFrame, horizons: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    """ΔV (€), ΔV% e log-retorno ln(V_{T+k}/V_T). V_T ≤ 0 é tratado como não avaliado."""
    out = df.copy()
    v0 = out["tm_value_T"].where(out["tm_value_T"] > 0)
    for k in horizons:
        vk = out[f"tm_value_T{k}"]
        out[f"delta_v_T{k}"] = vk - v0
        out[f"delta_v_pct_T{k}"] = (vk - v0) / v0 * 100
        out[f"log_return_T{k}"] = np.log(vk.where(vk > 0) / v0)
    return out


@dataclass(frozen=True)
class SampleRules:
    min_minutes: float = 90
    value_quantile: float = 0.90
    # "season": limiar p90 por temporada (neutraliza a inflação do mercado); None: limiar único.
    quantile_by: str | None = "season"


def value_thresholds(df: pd.DataFrame, rules: SampleRules) -> pd.Series:
    if rules.quantile_by is None:
        return pd.Series({"all": df["tm_value_T"].quantile(rules.value_quantile)})
    return df.groupby(rules.quantile_by)["tm_value_T"].quantile(rules.value_quantile)


@dataclass(frozen=True)
class AnalyticSample:
    universe: pd.DataFrame     # todo jogador–temporada de T, com as flags `passes_*` de cada etapa
    sample: pd.DataFrame       # observações que passam por todas as etapas
    funnel: pd.DataFrame       # contagens por etapa
    thresholds: pd.Series      # limiar p90 de V_T (por temporada, ou único)


def build_analytic_sample(raw: pd.DataFrame, rules: SampleRules = SampleRules()) -> AnalyticSample:
    """Aplica o funil sequencial da especificação ao dataset bruto do notebook 01.

    A entrada é jogador–clube–temporada; a saída, jogador–temporada. O universo inteiro de T é
    devolvido com as flags de cada etapa (`passes_*`), para que o notebook 02 inspecione o que cada
    filtro remove.
    """
    base = raw[~raw["is_lag_only"]]
    steps: list[tuple[str, int, int]] = [
        ("Registros FBref (jogador–clube–temporada)", len(base), base["fbref_player_id"].nunique())
    ]
    ps = add_value_targets(add_demographics(to_player_season(base)))
    ps["passes_match"] = ps["tm_player_id"].notna()
    ps["passes_outfield"] = ps["passes_match"] & (ps["position_group"] != "GK") & ~ps["is_goalkeeper"]
    ps["passes_minutes"] = ps["passes_outfield"] & (ps["fb_minutes"] >= rules.min_minutes)
    ps["passes_value"] = ps["passes_minutes"] & (ps["tm_value_T"] > 0)
    eligible = ps[ps["passes_value"]]
    thresholds = value_thresholds(eligible, rules)
    cap = ps[rules.quantile_by].map(thresholds) if rules.quantile_by else thresholds.iloc[0]
    ps["value_cap_T"] = cap
    ps["passes_p90"] = ps["passes_value"] & (ps["tm_value_T"] <= ps["value_cap_T"])

    labels = [
        ("Jogador–temporada", None),
        ("Pareados com o Transfermarkt", "passes_match"),
        ("Exclusão de goleiros", "passes_outfield"),
        (f"Minutagem ≥ {rules.min_minutes:g} min em T", "passes_minutes"),
        ("Com valor de mercado em T", "passes_value"),
        (f"Corte p{rules.value_quantile * 100:g} de V_T", "passes_p90"),
    ]
    for label, flag in labels:
        sub = ps if flag is None else ps[ps[flag]]
        steps.append((label, len(sub), sub["fbref_player_id"].nunique()))
    funnel = pd.DataFrame(steps, columns=["etapa", "observacoes", "atletas_unicos"])
    funnel["retencao_etapa_pct"] = funnel["observacoes"] / funnel["observacoes"].shift(1) * 100
    # Retenção acumulada medida a partir da unidade de análise (jogador–temporada).
    funnel["retencao_acumulada_pct"] = funnel["observacoes"] / funnel["observacoes"].iloc[1] * 100
    funnel.loc[0, ["retencao_etapa_pct", "retencao_acumulada_pct"]] = np.nan
    ps["in_sample"] = ps["passes_p90"]
    return AnalyticSample(ps, ps[ps["in_sample"]].reset_index(drop=True), funnel, thresholds)
