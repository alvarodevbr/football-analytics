"""Engenharia de features em T: agregação jogador–temporada das três fontes de desempenho, taxas
por 90 minutos (com encolhimento empírico-bayesiano), atributos demográficos, contexto de clube e
defasagens (T−1).

Todas as features usam apenas informação disponível até a data-âncora de T (30/jun do ano de
término da temporada). As distribuições a priori do encolhimento são estimadas no corte transversal
de cada temporada e posição, sem nenhum alvo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .sample import KEY, LEAGUE_LABELS
from .valuation import attach_values

# --------------------------------------------------------------------------------------------
# Colunas por fonte
# --------------------------------------------------------------------------------------------
FB_COUNTS: tuple[str, ...] = (
    "fb_games", "fb_games_starts", "fb_minutes", "fb_goals", "fb_assists", "fb_goals_pens",
    "fb_pens_made", "fb_pens_att", "fb_cards_yellow", "fb_cards_red", "fb_cards_yellow_red",
    "fb_shots", "fb_shots_on_target", "fb_games_complete", "fb_games_subs", "fb_unused_subs",
    "fb_on_goals_for", "fb_on_goals_against", "fb_plus_minus", "fb_fouls", "fb_fouled",
    "fb_offsides", "fb_crosses", "fb_interceptions", "fb_tackles_won", "fb_own_goals",
)
US_COUNTS: tuple[str, ...] = (
    "us_matches", "us_minutes", "us_goals", "us_xg", "us_np_goals", "us_np_xg", "us_assists", "us_xa",
    "us_shots", "us_key_passes", "us_xg_chain", "us_xg_buildup",
)
SS_COUNTS: tuple[str, ...] = (
    "ss_minutes_played", "ss_appearances", "ss_matches_started", "ss_goals", "ss_assists",
    "ss_total_shots", "ss_shots_on_target", "ss_big_chances_created", "ss_big_chances_missed",
    "ss_total_passes", "ss_accurate_passes", "ss_key_passes", "ss_accurate_long_balls",
    "ss_accurate_final_third_passes", "ss_accurate_crosses", "ss_successful_dribbles", "ss_touches",
    "ss_possession_lost", "ss_dispossessed", "ss_tackles", "ss_interceptions", "ss_clearances",
    "ss_dribbled_past", "ss_aerial_duels_won", "ss_ground_duels_won", "ss_total_duels_won",
    "ss_error_lead_to_shot", "ss_error_lead_to_goal", "ss_was_fouled", "ss_fouls",
    "ss_yellow_cards", "ss_red_cards",
)
# Taxas publicadas sem o denominador: agregadas pela média ponderada pelos minutos.
SS_RATES: tuple[str, ...] = (
    "ss_rating", "ss_accurate_long_balls_percentage", "ss_successful_dribbles_percentage",
    "ss_aerial_duels_won_percentage", "ss_ground_duels_won_percentage", "ss_total_duels_won_percentage",
)
# Sem cobertura histórica no Sofascore (notebook 02, seção 5): descartadas.
SS_DROPPED: tuple[str, ...] = (
    "ss_expected_goals", "ss_expected_assists", "ss_ball_recovery", "ss_outfielder_blocks",
)

PER90: dict[str, tuple[str, str]] = {
    # nome da feature: (contagem, minutos da mesma fonte)
    "goals_p90": ("fb_goals", "fb_minutes"),
    "assists_p90": ("fb_assists", "fb_minutes"),
    "npg_p90": ("fb_goals_pens", "fb_minutes"),
    "shots_p90": ("fb_shots", "fb_minutes"),
    "shots_on_target_p90": ("fb_shots_on_target", "fb_minutes"),
    "yellow_cards_p90": ("fb_cards_yellow", "fb_minutes"),
    "red_cards_p90": ("fb_cards_red", "fb_minutes"),
    "fouls_p90": ("fb_fouls", "fb_minutes"),
    "fouled_p90": ("fb_fouled", "fb_minutes"),
    "offsides_p90": ("fb_offsides", "fb_minutes"),
    "crosses_p90": ("fb_crosses", "fb_minutes"),
    "interceptions_p90": ("fb_interceptions", "fb_minutes"),
    "tackles_won_p90": ("fb_tackles_won", "fb_minutes"),
    "on_goals_against_p90": ("fb_on_goals_against", "fb_minutes"),
    "plus_minus_p90": ("fb_plus_minus", "fb_minutes"),
    "xg_p90": ("us_xg", "us_minutes"),
    "npxg_p90": ("us_np_xg", "us_minutes"),
    "xa_p90": ("us_xa", "us_minutes"),
    "xg_chain_p90": ("us_xg_chain", "us_minutes"),
    "xg_buildup_p90": ("us_xg_buildup", "us_minutes"),
    "key_passes_p90": ("ss_key_passes", "ss_minutes_played"),
    "big_chances_created_p90": ("ss_big_chances_created", "ss_minutes_played"),
    "big_chances_missed_p90": ("ss_big_chances_missed", "ss_minutes_played"),
    "passes_p90": ("ss_total_passes", "ss_minutes_played"),
    "accurate_passes_p90": ("ss_accurate_passes", "ss_minutes_played"),
    "long_balls_p90": ("ss_accurate_long_balls", "ss_minutes_played"),
    "final_third_passes_p90": ("ss_accurate_final_third_passes", "ss_minutes_played"),
    "accurate_crosses_p90": ("ss_accurate_crosses", "ss_minutes_played"),
    "dribbles_p90": ("ss_successful_dribbles", "ss_minutes_played"),
    "touches_p90": ("ss_touches", "ss_minutes_played"),
    "possession_lost_p90": ("ss_possession_lost", "ss_minutes_played"),
    "dispossessed_p90": ("ss_dispossessed", "ss_minutes_played"),
    "tackles_p90": ("ss_tackles", "ss_minutes_played"),
    "clearances_p90": ("ss_clearances", "ss_minutes_played"),
    "dribbled_past_p90": ("ss_dribbled_past", "ss_minutes_played"),
    "aerial_duels_won_p90": ("ss_aerial_duels_won", "ss_minutes_played"),
    "ground_duels_won_p90": ("ss_ground_duels_won", "ss_minutes_played"),
    "errors_p90": ("ss_errors", "ss_minutes_played"),
}

# --------------------------------------------------------------------------------------------
# Nacionalidade
# --------------------------------------------------------------------------------------------
_CONMEBOL = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru",
             "Uruguay", "Venezuela"}
_CONCACAF = {"Antigua and Barbuda", "Aruba", "Barbados", "Bermuda", "Canada", "Costa Rica", "Cuba",
             "Curacao", "Dominica", "Dominican Republic", "French Guiana", "Grenada", "Guadeloupe",
             "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Montserrat",
             "Panama", "Saint-Martin", "St. Kitts & Nevis", "St. Lucia", "St. Vincent & Grenadinen",
             "Suriname", "Trinidad and Tobago", "United States"}
_AFC = {"Afghanistan", "Australia", "Bangladesh", "China", "India", "Indonesia", "Iran", "Iraq", "Japan",
        "Jordan", "Korea, South", "Lebanon", "Palestine", "Philippines", "Saudi Arabia", "Syria", "Thailand",
        "United Arab Emirates", "Uzbekistan"}
_OFC = {"New Zealand", "New Caledonia"}
_UEFA = {"Albania", "Armenia", "Austria", "Belgium", "Bosnia-Herzegovina", "Bulgaria", "Croatia", "Cyprus",
         "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands", "Finland", "France", "Georgia",
         "Germany", "Greece", "Guernsey", "Hungary", "Iceland", "Ireland", "Isle of Man", "Israel", "Italy",
         "Jersey", "Kazakhstan", "Kosovo", "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco",
         "Montenegro", "Netherlands", "North Macedonia", "Northern Ireland", "Norway", "Poland", "Portugal",
         "Romania", "Russia", "San Marino", "Scotland", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
         "Switzerland", "Türkiye", "Ukraine", "United Kingdom", "Wales"}
# Os demais países presentes na base são da CAF; `confederation` devolve "CAF" só para eles.
_CAF = {"Algeria", "Angola", "Benin", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde",
        "Central African Republic", "Chad", "Comoros", "Congo", "Cote d'Ivoire", "DR Congo", "Egypt",
        "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Ghana", "Guinea", "Guinea-Bissau", "Kenya",
        "Liberia", "Libya", "Madagascar", "Mali", "Mauritania", "Morocco", "Mozambique", "Niger", "Nigeria",
        "Réunion", "Rwanda", "Sao Tome and Principe", "Senegal", "Sierra Leone", "South Africa",
        "Southern Sudan", "Tanzania", "The Gambia", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"}
CONFEDERATIONS: dict[str, set[str]] = {"UEFA": _UEFA, "CONMEBOL": _CONMEBOL, "CAF": _CAF, "AFC": _AFC,
                                       "CONCACAF": _CONCACAF, "OFC": _OFC}

EU27 = {"Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia",
        "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
        "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain",
        "Sweden",
        # territórios franceses: cidadania francesa
        "French Guiana", "Guadeloupe", "Martinique", "Réunion", "Saint-Martin", "New Caledonia"}
UK = {"England", "Scotland", "Wales", "Northern Ireland", "United Kingdom"}
# Até o fim do período de transição do Brexit (31/12/2020), britânicos tinham livre circulação na UE.
UK_EU_UNTIL_SEASON = "1920"

LEAGUE_COUNTRY: dict[str, set[str]] = {
    "ENG-Premier League": UK,
    "ESP-La Liga": {"Spain"},
    "ITA-Serie A": {"Italy"},
    "GER-Bundesliga": {"Germany"},
    "FRA-Ligue 1": {"France"},
}


def confederation(country: str | None) -> str | None:
    if not country:
        return None
    for name, members in CONFEDERATIONS.items():
        if country in members:
            return name
    return None


def nationality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Confederação da 1ª nacionalidade, passaporte comunitário e atleta do país da liga."""
    nats = df["tm_nationalities"].fillna("").str.split("|").apply(lambda xs: [x.strip() for x in xs if x.strip()])
    out = pd.DataFrame(index=df.index)
    out["confederation"] = nats.str[0].map(confederation)
    for conf in ("UEFA", "CONMEBOL", "CAF"):
        out[f"is_{conf.lower()}"] = (out["confederation"] == conf).astype(float)
    # agrupamento da especificação: AFC/CONCACAF (OFC incluída pelo tamanho ínfimo)
    out["is_afc_concacaf"] = out["confederation"].isin(["AFC", "CONCACAF", "OFC"]).astype(float)
    eu = nats.apply(lambda xs: any(x in EU27 for x in xs))
    uk = nats.apply(lambda xs: any(x in UK for x in xs))
    out["has_eu_passport"] = (eu | (uk & (df["season"] <= UK_EU_UNTIL_SEASON))).astype(float)
    out["n_nationalities"] = nats.str.len()
    out["is_domestic"] = [
        float(any(x in LEAGUE_COUNTRY.get(lg, set()) for x in xs)) for xs, lg in zip(nats, df["league"])
    ]
    missing = nats.str.len() == 0
    out.loc[missing, [c for c in out.columns if c != "confederation"]] = np.nan
    return out


# --------------------------------------------------------------------------------------------
# Agregação jogador–temporada
# --------------------------------------------------------------------------------------------
def _aggregate_source(raw: pd.DataFrame, ids: list[str], counts: tuple[str, ...], minutes: str,
                      rates: tuple[str, ...] = ()) -> pd.DataFrame:
    """Soma as contagens de uma fonte por jogador–temporada, contando cada linha da fonte uma vez.

    Um transferido pode estar casado com uma única linha agregada da fonte (compartilhada pelas
    duas linhas do FBref) ou com uma linha por clube; a deduplicação pelo id da fonte cobre os dois.
    """
    cols = [c for c in counts + rates if c in raw.columns]
    rows = raw.dropna(subset=[ids[0]]).drop_duplicates(KEY + ids)[KEY + ids + cols].copy()
    for rate in rates:
        rows[f"_{rate}_w"] = rows[rate] * rows[minutes]
    agg = rows.groupby(KEY).agg({**{c: "sum" for c in counts if c in rows},
                                 **{f"_{r}_w": "sum" for r in rates}, ids[0]: "first"})
    for rate in rates:
        agg[rate] = (agg.pop(f"_{rate}_w") / agg[minutes].where(agg[minutes] > 0))
    agg[f"{ids[0].split('_')[0]}_n_rows"] = rows.groupby(KEY).size()
    return agg.reset_index()


def aggregate_player_season(raw: pd.DataFrame) -> pd.DataFrame:
    """Jogador–temporada com contagens somadas entre clubes e taxas recalculadas.

    Os atributos não numéricos (clube, posição, dados do Transfermarkt) vêm do clube com mais
    minutos, como em `sample.to_player_season`.
    """
    raw = raw.copy()
    pct = raw["fb_minutes_pct"].where(raw["fb_minutes_pct"] > 0)
    raw["fb_possible_minutes"] = raw["fb_minutes"] / pct * 100
    raw["fb_ppg_x_games"] = raw["fb_points_per_game"] * raw["fb_games"]
    fb_sum = tuple(c for c in FB_COUNTS if c in raw.columns) + ("fb_possible_minutes", "fb_ppg_x_games")

    ordered = raw.sort_values("fb_minutes", ascending=False, na_position="last")
    us_cols = [c for c in raw.columns if c.startswith("us_") or c.startswith("understat_")]
    ss_cols = [c for c in raw.columns if c.startswith("ss_") or c.startswith("sofascore_")]
    main = ordered.drop_duplicates(KEY).drop(columns=list(fb_sum) + us_cols + ss_cols)
    fb = raw.groupby(KEY, as_index=False)[list(fb_sum)].sum(min_count=1)
    n = raw.groupby(KEY).size().rename("n_club_rows").reset_index()
    out = main.merge(fb, on=KEY).merge(n, on=KEY)
    out["fb_minutes_pct"] = out["fb_minutes"] / out["fb_possible_minutes"] * 100
    out["fb_points_per_game"] = out.pop("fb_ppg_x_games") / out["fb_games"].where(out["fb_games"] > 0)

    us = _aggregate_source(raw, ["understat_player_id", "understat_team_id"], US_COUNTS, "us_minutes")
    ss = _aggregate_source(raw, ["sofascore_player_id", "sofascore_team_id"], SS_COUNTS, "ss_minutes_played",
                           SS_RATES)
    out = out.merge(us, on=KEY, how="left").merge(ss, on=KEY, how="left")
    return out


# --------------------------------------------------------------------------------------------
# Encolhimento empírico-bayesiano
# --------------------------------------------------------------------------------------------
def _group_keys(groups: pd.DataFrame | pd.Series) -> pd.Series:
    """Chave única por grupo (temporada × posição); ausentes formam um grupo próprio."""
    if isinstance(groups, pd.Series):
        return groups.astype(str).fillna("NA")
    return groups.astype(str).fillna("NA").agg("|".join, axis=1)


def shrink_counts(counts: pd.Series, exposure: pd.Series, groups: pd.DataFrame | pd.Series,
                  min_exposure: float = 1.0) -> pd.Series:
    """Taxas por unidade de exposição com encolhimento Gamma–Poisson (Clayton e Kaldor, 1987;
    Marshall, 1991).

    Em cada grupo, a priori Gamma tem média μ = Σc/Σe e variância τ² estimada por momentos:
    τ² = s²_w − μ/ē, com s²_w a variância das taxas ponderada pela exposição e ē a exposição média.
    A estimativa a posteriori é (α + c)/(β + e), com α = μ²/τ² e β = μ/τ²: quem tem pouca exposição
    fica perto da média do grupo, quem tem muita fica perto da própria taxa. τ² ≤ 0 (nenhuma
    heterogeneidade além do ruído) leva ao encolhimento total. A priori é estimada com quem tem ao
    menos `min_exposure` (1 = 90 minutos, a população da amostra) e aplicada a todos com exposição;
    sem exposição, a taxa fica ausente.
    """
    c, e = counts.astype(float), exposure.astype(float)
    ok = c.notna() & e.notna() & (e > 0)
    out = pd.Series(np.nan, index=counts.index)
    keys = _group_keys(groups)
    for _, idx in keys[ok].groupby(keys[ok]).groups.items():
        cg, eg = c.loc[idx].clip(lower=0), e.loc[idx]
        fit = eg >= min_exposure
        mu = cg[fit].sum() / eg[fit].sum() if fit.sum() >= 3 else 0.0
        if mu <= 0:
            out.loc[idx] = cg / eg
            continue
        rate = cg[fit] / eg[fit]
        s2 = float((eg[fit] * (rate - mu) ** 2).sum() / eg[fit].sum())
        tau2 = max(s2 - mu / float(eg[fit].mean()), 1e-12)
        alpha, beta = mu ** 2 / tau2, mu / tau2
        out.loc[idx] = (alpha + cg) / (beta + eg)
    return out


def shrink_proportions(successes: pd.Series, attempts: pd.Series, groups: pd.DataFrame | pd.Series) -> pd.Series:
    """Proporções com encolhimento Beta–binomial, a priori estimada por momentos em cada grupo.

    κ = p̄(1 − p̄)/σ²_b − 1 é o tamanho da amostra a priori, com σ²_b a variância entre atletas
    (variância ponderada das proporções menos o ruído binomial p̄(1 − p̄)/n̄). A estimativa a
    posteriori é (s + κp̄)/(n + κ). Sem tentativas, devolve a média do grupo.
    """
    s_, n_ = successes.astype(float), attempts.astype(float)
    ok = s_.notna() & n_.notna() & (n_ >= 0)
    out = pd.Series(np.nan, index=successes.index)
    keys = _group_keys(groups)
    for _, idx in keys[ok].groupby(keys[ok]).groups.items():
        sg, ng = s_.loc[idx], n_.loc[idx]
        sg = sg.clip(lower=0).where(sg <= ng, ng)
        if ng.sum() <= 0:
            continue
        pbar = sg.sum() / ng.sum()
        has = ng > 0
        if not 0 < pbar < 1 or has.sum() < 3:
            out.loc[idx] = (sg / ng.where(ng > 0)).fillna(pbar)
            continue
        p = sg[has] / ng[has]
        var_w = float((ng[has] * (p - pbar) ** 2).sum() / ng[has].sum())
        var_b = var_w - pbar * (1 - pbar) / float(ng[has].mean())
        kappa = pbar * (1 - pbar) / var_b - 1 if var_b > 0 else 1e12
        kappa = max(kappa, 0.0)
        out.loc[idx] = (sg + kappa * pbar) / (ng + kappa)
    return out


def shrink_means(values: pd.Series, weights: pd.Series, groups: pd.DataFrame | pd.Series,
                 n_bins: int = 5, min_weight: float = 1.0) -> pd.Series:
    """Médias com encolhimento normal–normal: x̂ = μ + B(x − μ), B = τ²/(τ² + σ²/w).

    Em cada grupo, τ² (variância entre atletas) e σ² (ruído por unidade de peso) vêm dos momentos
    Var(x | w) = τ² + σ²/w, estimados por faixas de peso: a variância de x em cada quintil de `w` é
    regredida na média de 1/w do quintil, só com quem tem peso ≥ `min_weight` (a população da
    amostra). Se a variância não cai com o peso (σ² ≤ 0), o ruído é desprezível e não há
    encolhimento. Serve à nota média e às métricas com sinal (saldo de gols em campo, finalização
    acima do esperado), em que o modelo de Poisson não se aplica.
    """
    x, w = values.astype(float), weights.astype(float)
    ok = x.notna() & w.notna() & (w > 0)
    out = pd.Series(np.nan, index=values.index)
    keys = _group_keys(groups)
    for _, idx in keys[ok].groupby(keys[ok]).groups.items():
        xg, wg = x.loc[idx], w.loc[idx]
        fit = wg >= min_weight
        if fit.sum() < 5 * n_bins:
            out.loc[idx] = xg
            continue
        mu = float((wg[fit] * xg[fit]).sum() / wg[fit].sum())
        bins = pd.qcut(wg[fit].rank(method="first"), n_bins, labels=False)
        moments = pd.DataFrame({"var": xg[fit].groupby(bins).var(), "inv_w": (1 / wg[fit]).groupby(bins).mean()})
        design = np.column_stack([np.ones(len(moments)), moments["inv_w"].to_numpy()])
        (tau2, sigma2), *_ = np.linalg.lstsq(design, moments["var"].to_numpy(), rcond=None)
        tau2, sigma2 = max(float(tau2), 1e-12), max(float(sigma2), 0.0)
        shrink = tau2 / (tau2 + sigma2 / wg)
        out.loc[idx] = mu + shrink * (xg - mu)
    return out


# --------------------------------------------------------------------------------------------
# Taxas e eficiências
# --------------------------------------------------------------------------------------------
def _ratio(num: pd.Series, den: pd.Series, scale: float = 1.0) -> pd.Series:
    return num / den.where(den > 0) * scale


def _attempts_from_pct(successes: pd.Series, pct: pd.Series) -> pd.Series:
    """Tentativas implícitas numa taxa publicada sem denominador (sucessos / %); 0% com 0 sucessos → 0."""
    n = successes / (pct / 100).where(pct > 0)
    return n.where(pct > 0, np.where(successes.fillna(0) == 0, 0.0, np.nan)).where(pct.notna())


# Taxas por 90 de contagens compostas: (contagens somadas, minutos da mesma fonte).
COMPOSITE_PER90: dict[str, tuple[tuple[str, ...], str]] = {
    "tackles_interceptions_p90": (("ss_tackles", "ss_interceptions"), "ss_minutes_played"),
    "goals_assists_p90": (("fb_goals", "fb_assists"), "fb_minutes"),
    "xg_xa_p90": (("us_xg", "us_xa"), "us_minutes"),
}
# Métricas por 90 com sinal: (numerador, minutos).
SIGNED_PER90: tuple[str, ...] = ("plus_minus_p90", "finishing_overperf_p90")
# Percentuais: (sucessos, tentativas explícitas) ou (sucessos, percentual publicado pelo Sofascore).
PROPORTIONS: dict[str, tuple[str, str, bool]] = {
    "shot_accuracy_pct": ("fb_shots_on_target", "fb_shots", False),
    "shot_conversion_pct": ("fb_goals", "fb_shots", False),
    "pass_accuracy_pct": ("ss_accurate_passes", "ss_total_passes", False),
    "long_ball_accuracy_pct": ("ss_accurate_long_balls", "ss_accurate_long_balls_percentage", True),
    "dribble_success_pct": ("ss_successful_dribbles", "ss_successful_dribbles_percentage", True),
    "aerial_duels_won_pct": ("ss_aerial_duels_won", "ss_aerial_duels_won_percentage", True),
    "ground_duels_won_pct": ("ss_ground_duels_won", "ss_ground_duels_won_percentage", True),
    "duels_won_pct": ("ss_total_duels_won", "ss_total_duels_won_percentage", True),
}
# Lista das features encolhidas; cada uma tem uma versão crua `<nome>_raw` (papel `audit`).
SHRUNK_FEATURES: tuple[str, ...] = (
    *[k for k in PER90 if k not in SIGNED_PER90], *COMPOSITE_PER90, *SIGNED_PER90, *PROPORTIONS,
    "xg_per_shot", "sofascore_rating",
)


def performance_features(ps: pd.DataFrame, shrink: bool = True,
                         group_cols: tuple[str, ...] = ("season", "position_group")) -> pd.DataFrame:
    """Taxas por 90 minutos (minutos da própria fonte), eficiências e volume de participação.

    Com `shrink`, as taxas e eficiências passam pelo encolhimento empírico-bayesiano por temporada
    × posição, e as versões cruas ficam em colunas `<nome>_raw`. O volume (minutos, jogos,
    titularidade) não é encolhido: ele é a própria exposição.
    """
    ps = ps.copy()
    ps["ss_errors"] = ps["ss_error_lead_to_shot"].fillna(0) + ps["ss_error_lead_to_goal"].fillna(0)
    ps.loc[ps["ss_minutes_played"].isna(), "ss_errors"] = np.nan
    ps["_finishing_overperf"] = ps["us_np_goals"] - ps["us_np_xg"]
    groups = ps[list(group_cols)] if shrink and all(c in ps for c in group_cols) else None
    out = pd.DataFrame(index=ps.index)
    raw: dict[str, pd.Series] = {}

    def per90(name: str, count: pd.Series, minutes: pd.Series, signed: bool = False) -> None:
        raw[name] = _ratio(count, minutes, 90)
        if groups is None:
            out[name] = raw[name]
        elif signed:
            out[name] = shrink_means(raw[name], minutes / 90, groups)
        else:
            out[name] = shrink_counts(count, minutes / 90, groups)  # exposição em jogos de 90 min

    for name, (count, minutes) in PER90.items():
        per90(name, ps[count], ps[minutes], signed=name in SIGNED_PER90)
    for name, (counts, minutes) in COMPOSITE_PER90.items():
        per90(name, ps[list(counts)].sum(axis=1, min_count=len(counts)), ps[minutes])
    per90("finishing_overperf_p90", ps["_finishing_overperf"], ps["us_minutes"], signed=True)
    # eficiências
    for name, (succ, den, from_pct) in PROPORTIONS.items():
        attempts = _attempts_from_pct(ps[succ], ps[den]) if from_pct else ps[den]
        raw[name] = ps[den] if from_pct else _ratio(ps[succ], ps[den], 100)
        out[name] = raw[name] if groups is None else shrink_proportions(ps[succ], attempts, groups) * 100
    raw["xg_per_shot"] = _ratio(ps["us_xg"], ps["us_shots"])
    # o Sofascore grava nota 0 para quem não chegou a ser avaliado (poucos minutos): é ausência
    raw["sofascore_rating"] = ps["ss_rating"].where(ps["ss_rating"] > 0)
    if groups is None:
        out["xg_per_shot"], out["sofascore_rating"] = raw["xg_per_shot"], raw["sofascore_rating"]
    else:
        out["xg_per_shot"] = shrink_means(raw["xg_per_shot"], ps["us_shots"], groups)
        out["sofascore_rating"] = shrink_means(raw["sofascore_rating"], ps["ss_minutes_played"] / 90, groups)
    # volume e participação
    out["minutes"] = ps["fb_minutes"]
    out["minutes_share_pct"] = ps["fb_minutes_pct"]
    out["games"] = ps["fb_games"]
    out["starts_share_pct"] = _ratio(ps["fb_games_starts"], ps["fb_games"], 100)
    out["complete_games_share_pct"] = _ratio(ps["fb_games_complete"], ps["fb_games"], 100)
    out["team_points_per_game"] = ps["fb_points_per_game"]
    out["n_clubs_big5"] = ps["n_club_rows"]
    if groups is not None:
        out = pd.concat([out, pd.DataFrame({f"{name}_raw": raw[name] for name in SHRUNK_FEATURES})], axis=1)
    return out


CONTEXT_COLUMNS: dict[str, str] = {
    "league_rank_pct_T": "ctx_league_rank_pct_T", "league_gd_per_match_T": "ctx_league_gd_per_match_T",
    "promoted_T": "ctx_promoted_T", "relegated_T": "ctx_relegated_T", "europe_T": "ctx_europe_T",
    "club_elo_T": "ctx_club_elo_T", "club_elo_change_T": "ctx_club_elo_change_T",
}


def club_context_features(ps: pd.DataFrame) -> pd.DataFrame:
    """Contexto do clube principal em T (notebook 01, `club_context`): classificação, saldo de gols
    por jogo, promoção, rebaixamento, participação europeia e, se disponível, Elo."""
    out = pd.DataFrame(index=ps.index)
    for name, col in CONTEXT_COLUMNS.items():
        out[name] = ps[col].astype(float) if col in ps else np.nan
    return out


# --------------------------------------------------------------------------------------------
# Demografia, mercado e clube
# --------------------------------------------------------------------------------------------
def demographic_features(ps: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=ps.index)
    # idade decimal na âncora (as faixas etárias seguem em anos completos)
    age = (ps["age_decimal_T"] if "age_decimal_T" in ps else ps["age_T"]).astype(float)
    out["age"] = age
    out["age_sq"] = age ** 2
    height = ps["tm_height_cm"].astype(float)
    if "tm_profile_height_cm" in ps:
        height = height.fillna(ps["tm_profile_height_cm"].astype(float))
    out["height_cm"] = height
    foot = ps["tm_foot"]
    out["is_left_footed"] = (foot == "left").astype(float).where(foot.notna())
    out["is_two_footed"] = (foot == "both").astype(float).where(foot.notna())
    return out.join(nationality_features(ps))


def market_features(ps: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """Nível de preço do atleta e do clube em T.

    O valor do clube é a soma de V_T de todos os atletas do universo jogador–temporada que tiveram
    o clube como principal na temporada (inclui goleiros e o topo excluído pelo p90): a força
    financeira do elenco que o atleta disputa.
    """
    club = (universe[universe["tm_value_T"] > 0]
            .groupby(["season", "fbref_team_id"])["tm_value_T"].agg(club_value_T="sum", club_n_valued="size")
            .reset_index())
    club["club_value_rank_pct"] = club.groupby("season")["club_value_T"].rank(pct=True)
    lg = universe[["season", "fbref_team_id", "league"]].drop_duplicates(["season", "fbref_team_id"])
    club = club.merge(lg, on=["season", "fbref_team_id"], how="left")
    club["club_value_rank_in_league"] = club.groupby(["season", "league"])["club_value_T"].rank(ascending=False)
    merged = ps[["season", "fbref_team_id"]].merge(club.drop(columns="league"), on=["season", "fbref_team_id"],
                                                   how="left")
    merged.index = ps.index
    out = pd.DataFrame(index=ps.index)
    out["value_T"] = ps["tm_value_T"]
    out["log_value_T"] = np.log(ps["tm_value_T"].where(ps["tm_value_T"] > 0))
    out["log_club_value_T"] = np.log(merged["club_value_T"])
    out["value_share_of_club"] = ps["tm_value_T"] / merged["club_value_T"]
    out["club_value_rank_pct"] = merged["club_value_rank_pct"]
    out["club_value_rank_in_league"] = merged["club_value_rank_in_league"]
    return out


def value_lags(ps: pd.DataFrame, market_values: pd.DataFrame, max_staleness_days: int = 183) -> pd.DataFrame:
    """V_{T−1}, V_{T−2} pelo histórico completo do Transfermarkt (cobre quem não estava nas Big Five) e
    os log-retornos passados (momentum)."""
    lagged = attach_values(ps[["tm_player_id", "season"]], market_values, horizons=(-1, -2),
                           max_staleness_days=max_staleness_days)
    lagged.index = ps.index
    out = pd.DataFrame(index=ps.index)
    v0 = ps["tm_value_T"].where(ps["tm_value_T"] > 0)
    for k, sfx in ((1, "Tm1"), (2, "Tm2")):
        vk = lagged[f"tm_value_{sfx}"].where(lagged[f"tm_value_{sfx}"] > 0)
        out[f"value_lag{k}"] = vk
        out[f"log_return_lag{k}"] = np.log(v0 / vk)
    return out


def market_regime(universe: pd.DataFrame, market_values: pd.DataFrame,
                  max_staleness_days: int = 183) -> pd.DataFrame:
    """Regime de mercado em T: log-retorno médio (média aparada a 10%) entre T−1 e T de todos os
    atletas avaliados. A mediana é inadequada aqui: como o Transfermarkt revisa em degraus, ela é
    exatamente zero na maioria das temporadas.

    Substitui efeitos fixos de temporada, que não se aplicam a uma temporada de teste nunca vista
    no treino. Usa só V_T e V_{T−1}, conhecidos em T. Devolve uma linha por (season, league), com
    o regime das Big Five e o da liga.
    """
    lr = value_lags(universe, market_values, max_staleness_days)["log_return_lag1"]
    d = universe[["season", "league"]].assign(lr=lr)
    trimmed = lambda x: stats.trim_mean(x.dropna(), 0.1)  # noqa: E731
    big5 = d.groupby("season")["lr"].agg(trimmed).rename("market_log_return_lag1_T")
    league = d.groupby(["season", "league"])["lr"].agg(trimmed).rename("league_log_return_lag1_T").reset_index()
    return league.merge(big5.reset_index(), on="season")


MARKET_HISTORY: tuple[str, ...] = (
    "mkt_log_return_6m", "mkt_last_change", "mkt_days_since_change", "mkt_n_revisions_12m",
    "mkt_n_up_24m", "mkt_n_down_24m", "mkt_volatility_24m", "mkt_peak_gap",
)


def market_history(rows: pd.DataFrame, market_values: pd.DataFrame) -> pd.DataFrame:
    """Dinâmica das reavaliações do Transfermarkt até a âncora de T (e só até ela).

    Captura a ancoragem e o ajuste gradual dos avaliadores com mais resolução que o log-retorno
    anual: variação nos últimos seis meses, tamanho e idade da última revisão, frequência e
    direção das revisões em 24 meses, volatilidade e distância ao pico histórico do atleta.
    """
    from .config import Season

    mv = market_values.dropna(subset=["mv_value_eur"])
    mv = mv[mv["mv_value_eur"] > 0].assign(mv_date=lambda d: pd.to_datetime(d["mv_date"]))
    mv = mv.sort_values(["tm_player_id", "mv_date"])
    hist = {int(pid): (g["mv_date"].to_numpy(), np.log(g["mv_value_eur"].to_numpy()))
            for pid, g in mv.groupby("tm_player_id")}
    day = np.timedelta64(1, "D")
    records = []
    for pid, season in zip(rows["tm_player_id"], rows["season"]):
        rec = dict.fromkeys(MARKET_HISTORY, np.nan)
        if pd.notna(pid) and int(pid) in hist:
            anchor = np.datetime64(pd.Timestamp(Season(season).end_anchor))
            dates, logv = hist[int(pid)]
            keep = dates <= anchor
            dates, logv = dates[keep], logv[keep]
            if len(logv):
                change, change_dates = np.diff(logv), dates[1:]
                moved = np.nonzero(change)[0]
                six_months_ago = logv[dates <= anchor - 182 * day]
                recent = change[change_dates > anchor - 730 * day]
                rec.update({
                    "mkt_log_return_6m": logv[-1] - six_months_ago[-1] if len(six_months_ago) else np.nan,
                    "mkt_last_change": change[moved[-1]] if len(moved) else 0.0,
                    "mkt_days_since_change": float((anchor - change_dates[moved[-1]]) / day) if len(moved) else np.nan,
                    "mkt_n_revisions_12m": float((dates > anchor - 365 * day).sum()),
                    "mkt_n_up_24m": float((recent > 0).sum()),
                    "mkt_n_down_24m": float((recent < 0).sum()),
                    "mkt_volatility_24m": float(recent.std()) if len(recent) >= 2 else np.nan,
                    "mkt_peak_gap": logv[-1] - logv.max(),
                })
        records.append(rec)
    return pd.DataFrame(records, index=rows.index)


# Insumos do valor justo: desempenho, volume, demografia e contexto. Ficam fora as variáveis
# construídas a partir do próprio V_T (participação no elenco, momentum, valor do elenco).
FAIR_VALUE_EXCLUDE: frozenset[str] = frozenset({
    "log_value_T", "value_share_of_club", "log_return_lag1", "log_return_lag2", "log_club_value_T",
    "club_value_rank_pct", "club_value_rank_in_league", *MARKET_HISTORY,
})


def valuation_gap(panel: pd.DataFrame, features: list[str], n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Valor justo em T e o gap de avaliação ln(V_T) − valor justo (negativo = subavaliado).

    O valor justo é a previsão de ln(V_T) a partir do desempenho em T, estimada **dentro de cada
    temporada** (corte transversal: todos os V_T de T são públicos em T, sem informação futura) por
    validação cruzada em K partes. Assim, o gap de cada atleta vem de um modelo que não o viu.
    """
    from sklearn.model_selection import KFold
    from xgboost import XGBRegressor

    cols = [c for c in features if c not in FAIR_VALUE_EXCLUDE]
    X = panel[cols].astype(float).join(pd.get_dummies(panel[["league", "position_group"]], dtype=float))
    y = np.log(panel["value_T"].astype(float))
    fair = pd.Series(np.nan, index=panel.index)
    for _, idx in panel.groupby("season").groups.items():
        idx = np.asarray(list(idx))
        for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(idx):
            model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.7, max_bin=64, n_jobs=4, random_state=seed, verbosity=0)
            model.fit(X.loc[idx[train]], y.loc[idx[train]])
            fair.loc[idx[test]] = model.predict(X.loc[idx[test]])
    return pd.DataFrame({"fair_log_value_T": fair, "value_gap_T": y - fair}, index=panel.index)


LAG_PERFORMANCE: tuple[str, ...] = (
    "minutes", "minutes_share_pct", "sofascore_rating", "goals_assists_p90", "xg_xa_p90", "key_passes_p90",
    "dribbles_p90", "tackles_interceptions_p90", "aerial_duels_won_p90", "pass_accuracy_pct",
)


def performance_lags(panel: pd.DataFrame, cols: tuple[str, ...] = LAG_PERFORMANCE) -> pd.DataFrame:
    """Desempenho do mesmo atleta em T−1 nas Big Five (NaN se não atuou nelas)."""
    from .config import Season

    prev = panel[["fbref_player_id", "season", *cols]].copy()
    prev["season"] = prev["season"].map(lambda s: Season(s).offset(1).code)
    prev = prev.rename(columns={c: f"lag1_{c}" for c in cols})
    out = panel[["fbref_player_id", "season"]].merge(prev, on=["fbref_player_id", "season"], how="left")
    out.index = panel.index
    out = out.drop(columns=["fbref_player_id", "season"])
    out["has_big5_lag1"] = out[f"lag1_{cols[0]}"].notna().astype(float)
    out["delta_minutes_share_pct"] = panel["minutes_share_pct"] - out["lag1_minutes_share_pct"]
    out["delta_sofascore_rating"] = panel["sofascore_rating"] - out["lag1_sofascore_rating"]
    return out


def league_label(df: pd.DataFrame) -> pd.Series:
    return df["league"].map(LEAGUE_LABELS)
