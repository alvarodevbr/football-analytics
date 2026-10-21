"""Resolução de entidades entre FBref e as demais fontes (sem chave primária compartilhada).

Implementa o procedimento descrito em specs/verificacao_dados.md §5.2:

* **clubes** — atribuição 1-para-1 (algoritmo húngaro) dentro de cada liga-temporada, onde as
  duas fontes listam exatamente os mesmos 18–20 clubes; aliases manuais têm precedência;
* **jogadores** — similaridade composta Levenshtein (``token_sort_ratio``) + Jaro-Winkler, com
  ``token_set_ratio`` para nomes parciais/apelidos, *blocking* por liga-temporada-clube,
  restrição dura de ano de nascimento (homônimos) ou de minutagem (fontes sem data de
  nascimento) e atribuição 1-para-1 por algoritmo húngaro. Estágios posteriores recuperam
  empréstimos/transferências (fallback na liga), propagam o id pelo ``fbref_player_id``
  (estável entre temporadas) e aplicam overrides manuais.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist
from scipy.optimize import linear_sum_assignment

from .config import Season
from .logging_utils import get_logger
from .text import normalize_club, normalize_name

log = get_logger("entity_resolution")

_INVALID = -1.0e6
SURNAME_INITIAL_SCORE = 88.0


@dataclass(frozen=True)
class MatchParams:
    accept_score: float = 85.0
    # aceito abaixo de `accept_score` só com validador forte (mesmo ano de nascimento / minutagem)
    weak_accept_score: float = 80.0
    fallback_score: float = 90.0
    cross_season_score: float = 92.0
    # último par restante de um bloco, validado
    residual_min_score: float = 60.0
    max_birth_year_gap: int = 1
    minutes_tolerance_abs: float = 90.0
    minutes_tolerance_rel: float = 0.15
    minutes_penalty: float = 15.0


# ----------------------------------------------------------------------------------------------
# similaridade
# ----------------------------------------------------------------------------------------------
def name_similarity(a: str, b: str) -> float:
    """Similaridade 0–100 entre nomes já normalizados.

    ``0,5·Levenshtein(token_sort) + 0,5·Jaro-Winkler`` captura variações de grafia e ordem;
    ``0,92·token_set_ratio`` cobre o caso em que um nome é subconjunto do outro ("Gabriel" ⊂
    "Gabriel Magalhaes"), com desconto porque é o padrão mais sujeito a falso positivo.
    """
    return float(similarity_matrix([a], [b])[0, 0])


def similarity_matrix(left: list[str] | np.ndarray, right: list[str] | np.ndarray) -> np.ndarray:
    """`name_similarity` para todos os pares (vetorizado com ``rapidfuzz.process.cdist``)."""
    left, right = list(left), list(right)
    if not left or not right:
        return np.zeros((len(left), len(right)))
    sort = cdist(left, right, scorer=fuzz.token_sort_ratio, dtype=np.float32)
    jaro = cdist(left, right, scorer=JaroWinkler.normalized_similarity, dtype=np.float32) * 100
    subset = cdist(left, right, scorer=fuzz.token_set_ratio, dtype=np.float32) * 0.92
    scores = np.maximum(0.5 * sort + 0.5 * jaro, subset).astype(float)
    # Diminutivos/hipocorísticos ("max"/"maximilian", "dan"/"daniel", "gio"/"giovanni"): mesmo
    # sobrenome e mesma inicial do prenome valem SURNAME_INITIAL_SCORE.
    l_tokens, r_tokens = [x.split() for x in left], [y.split() for y in right]
    l_last = np.array([t[-1] if len(t) > 1 else "" for t in l_tokens])
    r_last = np.array([t[-1] if len(t) > 1 else "" for t in r_tokens])
    l_init = np.array([t[0][0] if len(t) > 1 else "" for t in l_tokens])
    r_init = np.array([t[0][0] if len(t) > 1 else "" for t in r_tokens])
    same = ((l_last[:, None] == r_last[None, :]) & (l_init[:, None] == r_init[None, :])
            & (l_last[:, None] != "") & (l_init[:, None] != ""))
    scores = np.where(same, np.maximum(scores, SURNAME_INITIAL_SCORE), scores)
    empty = (np.array([not x for x in left])[:, None]) | (np.array([not y for y in right])[None, :])
    scores[empty] = 0.0
    return scores


# ----------------------------------------------------------------------------------------------
# clubes
# ----------------------------------------------------------------------------------------------
def map_teams(
    left: pd.DataFrame,
    right: pd.DataFrame,
    aliases: dict[str, str] | None = None,
    low_confidence: float = 80.0,
) -> pd.DataFrame:
    """Mapeia clubes da fonte esquerda para a direita dentro de cada (league, season).

    Ambos os frames têm colunas ``league``, ``season``, ``team`` (valores únicos por bloco).
    Retorna ``league, season, team, target_team, score, method``. Pares com score abaixo de
    `low_confidence` (sem alias) são registrados em log para revisão.
    """
    aliases = aliases or {}
    records: list[dict[str, object]] = []
    left_blocks = left[["league", "season", "team"]].drop_duplicates().groupby(["league", "season"])
    right_groups = {k: g["team"].drop_duplicates().tolist()
                    for k, g in right[["league", "season", "team"]].groupby(["league", "season"])}
    for (league, season), group in left_blocks:
        sources = group["team"].tolist()
        targets = list(right_groups.get((league, season), []))
        if len(sources) != len(targets):
            log.warning("número de clubes difere entre fontes",
                        extra={"league": league, "season": season, "left": len(sources), "right": len(targets)})
        forced = {s: aliases[s] for s in sources if s in aliases and aliases[s] in targets}
        free_sources = [s for s in sources if s not in forced]
        free_targets = [t for t in targets if t not in forced.values()]
        for s, t in forced.items():
            records.append({"league": league, "season": season, "team": s, "target_team": t,
                            "score": 100.0, "method": "alias"})
        if free_sources and free_targets:
            scores = np.array([[max(fuzz.WRatio(normalize_club(s), normalize_club(t)),
                                    fuzz.WRatio(normalize_name(s), normalize_name(t)))
                                for t in free_targets] for s in free_sources], dtype=float)
            rows, cols = linear_sum_assignment(-scores)
            for r, c in zip(rows, cols):
                records.append({"league": league, "season": season, "team": free_sources[r],
                                "target_team": free_targets[c], "score": float(scores[r, c]),
                                "method": "assignment"})
        assigned = {r["team"] for r in records if r["league"] == league and r["season"] == season}
        for s in sources:
            if s not in assigned:
                records.append({"league": league, "season": season, "team": s, "target_team": None,
                                "score": np.nan, "method": "unmapped"})
    out = pd.DataFrame(records)
    weak = out[(out["method"] == "assignment") & (out["score"] < low_confidence)]
    for row in weak.itertuples():
        log.warning("mapeamento de clube com baixa confiança — confira/adicione alias",
                    extra={"league": row.league, "season": row.season, "team": row.team,
                           "target": row.target_team, "score": round(row.score, 1)})
    return out


# ----------------------------------------------------------------------------------------------
# jogadores
# ----------------------------------------------------------------------------------------------
def _assign_block(
    left: pd.DataFrame,
    right: pd.DataFrame,
    params: MatchParams,
    *,
    accept: float,
    allow_weak: bool,
    allow_residual: bool,
    require_exact_birth: bool,
    require_minutes: bool,
) -> list[tuple[int, int, float, str]]:
    """Atribuição 1-para-1 em um bloco. Retorna (índice esquerdo, índice direito, score, método)."""
    if left.empty or right.empty:
        return []
    ln, rn = left["name_norm"].to_numpy(), right["name_norm"].to_numpy()
    lb, rb = left["birth_year"].to_numpy(float)[:, None], right["birth_year"].to_numpy(float)[None, :]
    lm, rm = left["minutes"].to_numpy(float)[:, None], right["minutes"].to_numpy(float)[None, :]
    lm_alt = left["minutes_alt"].to_numpy(float)[:, None]

    scores = similarity_matrix(ln, rn)
    birth_known = ~(np.isnan(lb) | np.isnan(rb))
    gap = np.where(birth_known, np.abs(lb - rb), 0.0)
    birth_exact = birth_known & (gap == 0)
    def _close(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        tolerance = np.maximum(params.minutes_tolerance_abs, params.minutes_tolerance_rel * np.fmax(a, b))
        return np.abs(a - b) <= tolerance

    # fontes que agregam a temporada inteira de um jogador transferido comparam com o total
    minutes_ok = np.isnan(lm) | np.isnan(rm) | _close(lm, rm) | (~np.isnan(lm_alt) & _close(lm_alt, rm))
    # validador forte: mesmo ano de nascimento, ou (sem data de nascimento) minutagem compatível
    strong = birth_exact | (~birth_known & minutes_ok)

    scores = np.where(minutes_ok, scores, scores - params.minutes_penalty)
    invalid = gap > params.max_birth_year_gap
    if require_exact_birth:
        invalid |= ~birth_exact
    if require_minutes:
        invalid |= ~minutes_ok
    scores[invalid] = _INVALID

    rows, cols = linear_sum_assignment(-scores)
    accepted: list[tuple[int, int, float, str]] = []
    used_l, used_r = set(), set()
    for i, j in zip(rows, cols):
        s = scores[i, j]
        if s <= _INVALID / 2:
            continue
        if s >= accept:
            method = "exact" if ln[i] == rn[j] else "fuzzy"
        elif allow_weak and s >= params.weak_accept_score and strong[i, j]:
            method = "fuzzy_validated"
        else:
            continue
        accepted.append((int(left.index[i]), int(right.index[j]), float(s), method))
        used_l.add(i)
        used_r.add(j)

    if allow_residual:
        rest_l = [i for i in range(len(left)) if i not in used_l]
        rest_r = [j for j in range(len(right)) if j not in used_r]
        if len(rest_l) == 1 and len(rest_r) == 1:
            i, j = rest_l[0], rest_r[0]
            if scores[i, j] >= params.residual_min_score and strong[i, j]:
                accepted.append((int(left.index[i]), int(right.index[j]), float(scores[i, j]), "residual"))
    return accepted


def _prepare(df: pd.DataFrame, name_col: str, birth_col: str | None, minutes_col: str | None,
             minutes_alt_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out["name_norm"] = out[name_col].map(normalize_name)
    out["birth_year"] = out[birth_col].astype(float) if birth_col else np.nan
    out["minutes"] = out[minutes_col].astype(float) if minutes_col else np.nan
    out["minutes_alt"] = out[minutes_alt_col].astype(float) if minutes_alt_col else np.nan
    return out


def match_players(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    left_team: str,
    right_team: str,
    right_id: str,
    left_birth: str | None = None,
    right_birth: str | None = None,
    left_minutes: str | None = None,
    right_minutes: str | None = None,
    left_minutes_total: str | None = None,
    params: MatchParams = MatchParams(),
    cross_season: bool = False,
    share_season_aggregate: bool = False,
) -> pd.DataFrame:
    """Casa cada linha de `left` (FBref) com no máximo uma linha de `right`.

    Os índices dos dois frames precisam ser únicos (são devolvidos como ponte para o join).
    `left_team` já deve conter o nome do clube **na nomenclatura da fonte direita**.

    Estágios:
      1. ``(league, season, clube)`` — score ≥ `accept_score`, ou ≥ `weak_accept_score` com
         validador forte, e o par residual do bloco;
      2. ``(league, season)`` — linhas restantes contra todo o elenco da liga (empréstimos,
         inscrições tardias), com validador forte obrigatório e score ≥ `fallback_score`;
      3. propagação do id direito já associado ao mesmo ``fbref_player_id`` em outra temporada;
      3b. (opcional, `share_season_aggregate`) linhas do mesmo jogador-temporada em outro clube
         reutilizam a linha agregada já casada (fontes que não separam por clube);
      4. (opcional, `cross_season`) temporadas ±1 em qualquer liga, para fontes de cadastro
         (Transfermarkt), com ano de nascimento idêntico e score ≥ `cross_season_score`.

    Retorna ``left_index, right_index, right_id, match_score, match_method`` (uma linha por
    linha de `left`; não casadas ficam com NaN e método ``unmatched``).
    """
    has_birth = left_birth is not None and right_birth is not None
    has_minutes = left_minutes is not None and right_minutes is not None
    lp = _prepare(left, left_name, left_birth if has_birth else None, left_minutes if has_minutes else None,
                  left_minutes_total if has_minutes else None)
    rp = _prepare(right, right_name, right_birth if has_birth else None, right_minutes if has_minutes else None)

    result: dict[int, tuple[int, float, str]] = {}
    consumed: set[int] = set()

    # estágio 1 ---------------------------------------------------------------------------------
    right_blocks = {k: g for k, g in rp.groupby(["league", "season", right_team])}
    for key, block in lp.groupby(["league", "season", left_team]):
        candidates = right_blocks.get(key)
        if candidates is None:
            continue
        for li, ri, score, method in _assign_block(
                block, candidates, params, accept=params.accept_score, allow_weak=True,
                allow_residual=True, require_exact_birth=False, require_minutes=False):
            result[li] = (ri, score, method)
            consumed.add(ri)

    # estágio 2 ---------------------------------------------------------------------------------
    pending = lp[~lp.index.isin(result)]
    free_right = rp[~rp.index.isin(consumed)]
    right_by_ls = {k: g for k, g in free_right.groupby(["league", "season"])}
    for key, block in pending.groupby(["league", "season"]):
        candidates = right_by_ls.get(key)
        if candidates is None:
            continue
        for li, ri, score, _ in _assign_block(
                block, candidates, params, accept=params.fallback_score, allow_weak=False,
                allow_residual=False, require_exact_birth=has_birth, require_minutes=has_minutes and not has_birth):
            result[li] = (ri, score, "league_fallback")
            consumed.add(ri)

    # estágio 3: propagação pelo fbref_player_id -------------------------------------------------
    matched = pd.DataFrame(
        [(li, rp.at[ri, right_id]) for li, (ri, _, _) in result.items()], columns=["li", "rid"])
    if not matched.empty:
        matched["fbref_player_id"] = lp.loc[matched["li"], "fbref_player_id"].to_numpy()
        majority = matched.groupby("fbref_player_id")["rid"].agg(lambda s: s.value_counts().index[0])
        pending = lp[~lp.index.isin(result) & lp["fbref_player_id"].isin(majority.index)]
        free_right = rp[~rp.index.isin(consumed)]
        for li, row in pending.iterrows():
            rid = majority[row["fbref_player_id"]]
            same = free_right[(free_right[right_id] == rid) & (free_right["season"] == row["season"])
                              & (free_right["league"] == row["league"])]
            if not same.empty:
                ri = int(same.index[0])
                consumed.add(ri)
                free_right = free_right.drop(index=ri)
                result[li] = (ri, np.nan, "propagated")
            elif cross_season:
                # cadastro: o id vale mesmo sem linha de elenco nesta temporada
                anyrow = rp[rp[right_id] == rid]
                result[li] = (int(anyrow.index[0]), np.nan, "propagated")

    # estágio 3b: linha agregada da temporada (jogador transferido) ------------------------------
    # Understat/Sofascore publicam uma única linha por jogador-temporada mesmo quando o FBref tem
    # uma por clube: a mesma linha direita é compartilhada pelas linhas do mesmo jogador-temporada.
    if share_season_aggregate:
        owner = {(lp.at[li, "fbref_player_id"], lp.at[li, "season"]): ri for li, (ri, _, _) in result.items()}
        pending = lp[~lp.index.isin(result)]
        for li, row in pending.iterrows():
            ri = owner.get((row["fbref_player_id"], row["season"]))
            if ri is not None:
                result[li] = (ri, np.nan, "season_aggregate")

    # estágio 4: temporadas vizinhas (somente cadastro) ------------------------------------------
    if cross_season and has_birth:
        pending = lp[~lp.index.isin(result)]
        for season, block in pending.groupby("season"):
            neighbours = {Season(season).offset(-1).code, Season(season).offset(1).code}
            candidates = rp[rp["season"].isin(neighbours)].drop_duplicates(right_id)
            for li, ri, score, _ in _assign_block(
                    block, candidates, params, accept=params.cross_season_score, allow_weak=False,
                    allow_residual=False, require_exact_birth=True, require_minutes=False):
                result[li] = (ri, score, "cross_season")

    out = pd.DataFrame({"left_index": lp.index})
    out["right_index"] = out["left_index"].map(lambda i: result[i][0] if i in result else np.nan)
    out["match_score"] = out["left_index"].map(lambda i: result[i][1] if i in result else np.nan)
    out["match_method"] = out["left_index"].map(lambda i: result[i][2] if i in result else "unmatched")
    out[right_id] = out["right_index"].map(lambda ri: rp.at[int(ri), right_id] if pd.notna(ri) else np.nan)
    return out


def resolve_id_conflicts(
    df: pd.DataFrame, right_id: str, weight_col: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Força um único `right_id` por ``fbref_player_id`` (o mais frequente, ponderado por minutos).

    Linhas divergentes são reatribuídas e marcadas em ``id_conflict``; retorna também o relatório
    dos conflitos para revisão manual.
    """
    out = df.copy()
    out["id_conflict"] = False
    known = out[out[right_id].notna()]
    n_ids = known.groupby("fbref_player_id")[right_id].nunique()
    conflicted = n_ids[n_ids > 1].index
    if len(conflicted) == 0:
        return out, pd.DataFrame(columns=["fbref_player_id", right_id, "weight"])
    weights = known[known["fbref_player_id"].isin(conflicted)].assign(
        weight=lambda d: d[weight_col].fillna(0) + 1 if weight_col else 1.0)
    report = weights.groupby(["fbref_player_id", right_id])["weight"].sum().reset_index()
    winners = report.sort_values("weight", ascending=False).drop_duplicates("fbref_player_id")
    winner_map = dict(zip(winners["fbref_player_id"], winners[right_id]))
    mask = out["fbref_player_id"].isin(conflicted) & out[right_id].notna()
    changed = mask & (out[right_id] != out["fbref_player_id"].map(winner_map))
    out.loc[changed, right_id] = out.loc[changed, "fbref_player_id"].map(winner_map)
    out.loc[changed, "match_method"] = "conflict_resolved"
    out.loc[mask, "id_conflict"] = True
    log.warning("ids conflitantes resolvidos por maioria",
                extra={"right_id": right_id, "players": len(conflicted), "rows_changed": int(changed.sum())})
    return out, report


def load_aliases(path: Path, source: str) -> dict[str, str]:
    """Aliases de clube ``source,team,target_team`` (CSV versionado em ``data/external``)."""
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    table = table[table["source"] == source]
    return dict(zip(table["team"], table["target_team"]))


def load_player_overrides(path: Path) -> pd.DataFrame:
    """Overrides manuais ``fbref_player_id,tm_player_id,note``."""
    if not path.exists():
        return pd.DataFrame(columns=["fbref_player_id", "tm_player_id", "note"])
    return pd.read_csv(path, dtype={"fbref_player_id": str, "tm_player_id": "Int64", "note": str})
