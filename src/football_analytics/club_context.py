"""Contexto do clube em T: classificação final, saldo de gols, promoção, rebaixamento, participação
europeia e, se disponível, Elo.

Fontes:
* classificação e saldo de gols: resultados das partidas no Understat (3 pontos por vitória, 1 por
  empate; ordem por pontos por jogo, saldo por jogo e gols pró por jogo, o critério que a Ligue 1
  adotou em 2019/20, interrompida pela pandemia). Punições administrativas de pontos não entram;
* promoção e rebaixamento: presença do clube nas Big Five do FBref em T−1 e T+1 e, para a temporada
  seguinte à última coletada, nas partidas do Understat;
* participação europeia: listas de clubes das competições UEFA no Sofascore, inclusive as fases
  qualificatórias, pelo id do clube no Sofascore;
* Elo: ClubElo, opcional (a API respondia HTTP 502 em out/2026).

O Transfermarkt seria a fonte mais direta, mas bloqueou o acesso com captcha durante a coleta de
out/2026. As fontes acima não exigem pareamento difuso novo: o FBref e o Sofascore já estão ligados
ao painel pelos ids dos clubes, e o Understat pelo mapeamento de clubes da resolução de entidades.

Todas as variáveis são conhecidas na data-âncora de T (30/jun do ano de término da temporada): a
classificação, a promoção e o rebaixamento são decididos até o fim da temporada, inclusive os
play-offs de maio e junho (ler o pertencimento de T+1 só recupera um fato consumado na âncora), e a
participação europeia é a da própria temporada T. A vaga europeia para T+1 fica de fora de
propósito: parte dela (play-offs da Liga Europa e da Liga Conferência) só se decide em agosto,
depois da âncora.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date

import numpy as np
import pandas as pd

from . import entity_resolution as er
from .config import Season
from .logging_utils import get_logger

log = get_logger("club_context")

# (nome, id do torneio no Sofascore, nível, primeira temporada disputada)
UEFA_TOURNAMENTS: tuple[tuple[str, int, int, str], ...] = (
    ("Liga dos Campeões", 7, 3, "0001"),
    ("Liga Europa", 679, 2, "0001"),
    ("Liga Conferência", 17015, 1, "2122"),
)
EUROPE_LABELS = {0: "nenhuma", 1: "Liga Conferência", 2: "Liga Europa", 3: "Liga dos Campeões"}

CONTEXT_FEATURES: tuple[str, ...] = (
    "league_rank_pct_T", "league_gd_per_match_T", "promoted_T", "relegated_T", "europe_T",
    "club_elo_T", "club_elo_change_T",
)
ELO_FEATURES: tuple[str, ...] = ("club_elo_T", "club_elo_change_T")
ELO_COLUMNS = ["team", "league", "elo", "season", "kind"]


def tournaments_for(season: str) -> list[tuple[str, int, int]]:
    """Competições UEFA disputadas na temporada (a Liga Conferência começou em 2021/22)."""
    return [(name, tid, level) for name, tid, level, first in UEFA_TOURNAMENTS if season >= first]


def clubelo_dates(season: str) -> dict[str, date]:
    """Datas de leitura do ClubElo: meio da temporada (pertencimento à liga), início e âncora."""
    s = Season(season)
    return {"mid": date(s.start_year + 1, 1, 15), "start": date(s.start_year, 8, 1), "anchor": s.end_anchor}


# ----------------------------------------------------------------------------------------------
# blocos do contexto
# ----------------------------------------------------------------------------------------------
def standings(schedule: pd.DataFrame) -> pd.DataFrame:
    """Classificação final de cada liga-temporada a partir dos resultados das partidas.

    `schedule`: ``league, season, home_team, away_team, home_goals, away_goals`` e ``is_result``.
    Devolve uma linha por clube com jogos, pontos, saldo, posição (por pontos por jogo, saldo por
    jogo e gols pró por jogo) e a posição normalizada (0 = campeão, 1 = último).
    """
    played = schedule[schedule["is_result"].astype(bool)]
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        gf, ga = played[f"{side}_goals"].astype(float), played[f"{other}_goals"].astype(float)
        sides.append(pd.DataFrame({"league": played["league"], "season": played["season"], "team": played[f"{side}_team"],
                                   "gf": gf, "ga": ga, "points": np.select([gf > ga, gf == ga], [3, 1], 0)}))
    t = (pd.concat(sides).groupby(["league", "season", "team"], as_index=False)
         .agg(matches=("points", "size"), points=("points", "sum"), gf=("gf", "sum"), ga=("ga", "sum")))
    t["goal_diff"] = t["gf"] - t["ga"]
    t["ppm"], t["gdpm"], t["gfpm"] = t["points"] / t["matches"], t["goal_diff"] / t["matches"], t["gf"] / t["matches"]
    t = t.sort_values(["league", "season", "ppm", "gdpm", "gfpm"], ascending=[True, True, False, False, False])
    t["rank"] = t.groupby(["league", "season"]).cumcount() + 1
    n = t.groupby(["league", "season"])["rank"].transform("max")
    t["league_rank_pct_T"] = (t["rank"] - 1) / (n - 1)
    t["league_gd_per_match_T"] = t["gdpm"]
    return t[["league", "season", "team", "matches", "points", "goal_diff", "rank", "league_rank_pct_T",
              "league_gd_per_match_T"]].reset_index(drop=True)


def membership_flags(clubs: pd.DataFrame, seasons: Iterable[str]) -> pd.DataFrame:
    """Promoção e rebaixamento pelo pertencimento às Big Five em T−1 e T+1.

    `clubs` tem ``season`` e ``club`` (um clube das Big Five por linha). Para cada temporada de
    `seasons`, `promoted_T` = 1 se o clube não estava em T−1 e `relegated_T` = 1 se não está em T+1;
    NaN quando a temporada vizinha não foi coletada.
    """
    members = clubs.groupby("season")["club"].apply(set)
    rows = []
    for season in seasons:
        if season not in members:
            continue
        prev, nxt = Season(season).offset(-1).code, Season(season).offset(1).code
        for club in sorted(members[season]):
            rows.append({
                "season": season, "club": club,
                "promoted_T": float(club not in members[prev]) if prev in members else np.nan,
                "relegated_T": float(club not in members[nxt]) if nxt in members else np.nan,
            })
    return pd.DataFrame(rows, columns=["season", "club", "promoted_T", "relegated_T"])


def europe_levels(participants: pd.DataFrame) -> pd.DataFrame:
    """Maior nível de competição UEFA disputado por clube-temporada (3 = Liga dos Campeões).

    `participants`: ``season, club, level``. Um clube que caiu da Liga dos Campeões para a Liga
    Europa aparece nas duas listas e fica com 3.
    """
    if participants.empty:
        return pd.DataFrame(columns=["season", "club", "europe_T"])
    out = participants.groupby(["season", "club"], as_index=False)["level"].max()
    return out.rename(columns={"level": "europe_T"})


def read_clubelo(seasons: Iterable[str], read_by_date: Callable[[str], pd.DataFrame]) -> pd.DataFrame:
    """Tabela longa do ClubElo (season, kind, team, league, elo) nas três datas de cada temporada.

    Uma falha de conexão devolve a tabela vazia (o Elo é opcional; ver o docstring do módulo).
    A primeira data funciona como sonda: se ela falhar, as demais não são tentadas.
    """
    frames = []
    try:
        for season in seasons:
            for kind, when in clubelo_dates(season).items():
                df = read_by_date(when.isoformat()).reset_index()
                frames.append(df[["team", "league", "elo"]].assign(season=season, kind=kind))
    except (ConnectionError, OSError) as exc:
        log.warning("ClubElo indisponível; contexto segue sem Elo", extra={"error": str(exc)[:200]})
        return pd.DataFrame(columns=ELO_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def map_clubelo(fb_teams: pd.DataFrame, elo: pd.DataFrame, aliases: dict[str, str] | None = None) -> pd.DataFrame:
    """Clubes do FBref → nomes do ClubElo, dentro de cada liga-temporada (lista do meio da temporada)."""
    if elo.empty:
        return pd.DataFrame({c: pd.Series(dtype=float if c == "score" else object)
                             for c in ("league", "season", "team", "target_team", "score", "method")})
    right = elo[(elo["kind"] == "mid") & elo["league"].isin(fb_teams["league"].unique())]
    return er.map_teams(fb_teams[["league", "season", "team"]],
                        right[["league", "season", "team"]].drop_duplicates(), aliases)


def build_club_context(
    fb_teams: pd.DataFrame,
    understat_map: pd.DataFrame,
    table: pd.DataFrame,
    membership: pd.DataFrame,
    europe: pd.DataFrame,
    elo: pd.DataFrame,
    elo_map: pd.DataFrame,
) -> pd.DataFrame:
    """Uma linha por (season, fbref_team_id) com as variáveis de contexto do clube em T.

    `fb_teams`: league, season, team (nome FBref), fbref_team_id. `understat_map` e `elo_map` são
    saídas de `entity_resolution.map_teams` (team → target_team). `membership` e `europe` já vêm
    indexados pelo id do clube no FBref (coluna ``club``).
    """
    ctx = fb_teams.merge(understat_map[["league", "season", "team", "target_team"]], on=["league", "season", "team"],
                         how="left")
    ctx = (ctx.merge(table.rename(columns={"team": "target_team"}), on=["league", "season", "target_team"], how="left")
           .rename(columns={"target_team": "understat_team", "rank": "league_rank_T", "points": "league_points_T"}))
    ctx = (ctx.merge(membership.rename(columns={"club": "fbref_team_id"}), on=["season", "fbref_team_id"], how="left")
           .merge(europe.rename(columns={"club": "fbref_team_id"}), on=["season", "fbref_team_id"], how="left"))
    ctx["europe_T"] = ctx["europe_T"].fillna(0).astype(float)

    names = elo_map[["league", "season", "team", "target_team"]].rename(columns={"target_team": "clubelo_team"})
    ctx = ctx.merge(names, on=["league", "season", "team"], how="left")
    if elo.empty:
        ctx[list(ELO_FEATURES)] = np.nan
    else:
        wide = elo.pivot_table(index=["season", "team"], columns="kind", values="elo").reset_index()
        ctx = ctx.merge(wide.rename(columns={"team": "clubelo_team"}), on=["season", "clubelo_team"], how="left")
        ctx["club_elo_T"] = ctx["anchor"]
        ctx["club_elo_change_T"] = ctx["anchor"] - ctx["start"]
    keep = ["season", "league", "team", "fbref_team_id", "understat_team", "clubelo_team", "league_rank_T",
            "league_points_T", *CONTEXT_FEATURES]
    out = ctx[keep].rename(columns={"team": "fb_team"})
    if out.duplicated(["season", "fbref_team_id"]).any():
        raise ValueError("contexto de clube duplicado por (season, fbref_team_id)")
    return out.reset_index(drop=True)


def coverage(ctx: pd.DataFrame) -> pd.DataFrame:
    """Fração de clubes com cada variável preenchida, por temporada."""
    return ctx.groupby("season")[list(CONTEXT_FEATURES)].apply(lambda d: d.notna().mean())
