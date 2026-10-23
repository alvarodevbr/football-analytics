"""Etapas do notebook 01 como funções: cada uma lê/grava ``data/interim`` e pode ser reexecutada.

O notebook apenas encadeia estas funções e exibe os diagnósticos; toda a lógica testável vive aqui
e nos módulos de fonte.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import club_context as cc
from . import entity_resolution as er
from . import fbref, sofascore, transfermarkt, understat
from .config import BIG5_LEAGUES, ExtractionConfig, Season
from .http import Throttle, make_session
from .io import load_table, save_table
from .logging_utils import get_logger
from .valuation import attach_values

log = get_logger("pipeline")


def interim(cfg: ExtractionConfig, name: str) -> Path:
    return cfg.paths.interim / f"{name}.parquet"


# ----------------------------------------------------------------------------------------------
# 1. FBref
# ----------------------------------------------------------------------------------------------
def run_fbref(cfg: ExtractionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela jogador-temporada-clube do FBref + diagnóstico de preenchimento das páginas Opta."""
    fetcher = fbref.FBrefFetcher(cfg)
    # todas as competições das Big 5 são reconhecidas; o recorte por `cfg.leagues` vem depois
    comp_to_league = {lg.fbref_comp: lg.key for lg in BIG5_LEAGUES} | {"Fußball-Bundesliga": "GER-Bundesliga"}
    try:
        table = fbref.build_player_season_table(fetcher, cfg.seasons, cfg.fbref_pages, comp_to_league,
                                                cfg.league_keys)
        probe_season = Season(cfg.fbref_probe_season)
        fill = []
        for page in cfg.fbref_empty_probe_pages:
            rates = fbref.column_fill_rates(fbref.parse_player_table(fetcher.fetch(probe_season, page), page))
            fill.append(pd.DataFrame({"page": page, "season": probe_season.code, "column": rates.index,
                                      "fill_rate": rates.to_numpy()}))
    finally:
        fetcher.close()
    fill_rates = pd.concat(fill, ignore_index=True) if fill else pd.DataFrame()
    save_table(table, interim(cfg, "fbref_player_season"), source="FBref Big 5 combined pages (soccerdata fetch)")
    save_table(fill_rates, interim(cfg, "fbref_opta_fill_rates"), source="FBref páginas avançadas (diagnóstico)")
    return table, fill_rates


# ----------------------------------------------------------------------------------------------
# 2–3. Understat e Sofascore
# ----------------------------------------------------------------------------------------------
def run_understat(cfg: ExtractionConfig) -> pd.DataFrame:
    df = understat.read_player_seasons(cfg)
    save_table(df, interim(cfg, "understat_player_season"), source="understat.com via soccerdata")
    return df


def run_sofascore(cfg: ExtractionConfig) -> pd.DataFrame:
    collector = sofascore.SofascoreCollector(cfg)
    df = collector.collect(cfg.leagues, cfg.seasons)
    save_table(df, interim(cfg, "sofascore_player_season"),
               source="sofascore.com API /unique-tournament/.../statistics (por clube)")
    return df


# ----------------------------------------------------------------------------------------------
# 4. Transfermarkt — elencos
# ----------------------------------------------------------------------------------------------
def tm_client(cfg: ExtractionConfig) -> transfermarkt.TransfermarktClient:
    return transfermarkt.TransfermarktClient(
        make_session(cfg.user_agent),
        Throttle(cfg.tm_delay_seconds, cfg.request_jitter_seconds),
        cfg.paths.transfermarkt,
    )


def run_tm_squads(cfg: ExtractionConfig) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    client = tm_client(cfg)
    squads, failures = transfermarkt.collect_squads(
        client,
        [(lg.key, lg.tm_slug, lg.tm_competition_id) for lg in cfg.leagues],
        [(s.code, s.tm_saison_id) for s in cfg.seasons],
    )
    save_table(squads, interim(cfg, "tm_squads"), source="transfermarkt.com /kader/ (plus/1)")
    if failures:
        log.warning("clubes-temporada sem elenco — reexecute a etapa", extra={"n": len(failures)})
    return squads, failures


# ----------------------------------------------------------------------------------------------
# 5. Resolução de entidades
# ----------------------------------------------------------------------------------------------
@dataclass
class MatchOutputs:
    team_maps: pd.DataFrame
    tm: pd.DataFrame
    understat: pd.DataFrame
    sofascore: pd.DataFrame
    tm_conflicts: pd.DataFrame


def _team_frame(df: pd.DataFrame, team_col: str) -> pd.DataFrame:
    return df[["league", "season", team_col]].drop_duplicates().rename(columns={team_col: "team"})


def _apply_team_map(fb: pd.DataFrame, team_map: pd.DataFrame) -> pd.Series:
    lookup = {(r.league, r.season, r.team): r.target_team for r in team_map.itertuples()}
    return pd.Series([lookup.get(k) for k in zip(fb["league"], fb["season"], fb["fb_team"])], index=fb.index)


def run_entity_resolution(
    cfg: ExtractionConfig,
    fb: pd.DataFrame,
    squads: pd.DataFrame,
    us: pd.DataFrame,
    ss: pd.DataFrame,
    params: er.MatchParams = er.MatchParams(),
) -> MatchOutputs:
    aliases_path = cfg.paths.external / "team_aliases.csv"
    fb = fb.reset_index(drop=True)
    left_teams = _team_frame(fb, "fb_team")
    maps = {
        "transfermarkt": er.map_teams(left_teams, _team_frame(squads, "tm_team"),
                                      er.load_aliases(aliases_path, "transfermarkt")),
        "understat": er.map_teams(left_teams, _team_frame(us, "us_team"), er.load_aliases(aliases_path, "understat")),
        "sofascore": er.map_teams(left_teams, _team_frame(ss, "ss_team"), er.load_aliases(aliases_path, "sofascore")),
    }
    team_maps = pd.concat([m.assign(source=k) for k, m in maps.items()], ignore_index=True)

    # --- Transfermarkt (cadastro: ano de nascimento como validador, propagação entre temporadas)
    left = fb.assign(_tm_team=_apply_team_map(fb, maps["transfermarkt"]))
    right = squads.reset_index(drop=True).assign(
        tm_birth_year=lambda d: pd.to_datetime(d["tm_birth_date"]).dt.year)
    tm_match = er.match_players(
        left, right, left_name="fb_player", right_name="tm_player", left_team="_tm_team", right_team="tm_team",
        right_id="tm_player_id", left_birth="fb_birth_year", right_birth="tm_birth_year",
        params=params, cross_season=True)
    tm_match["fbref_player_id"] = fb.loc[tm_match["left_index"], "fbref_player_id"].to_numpy()
    tm_match["minutes"] = fb.loc[tm_match["left_index"], "fb_minutes"].to_numpy()
    tm_match["season"] = fb.loc[tm_match["left_index"], "season"].to_numpy()
    tm_match, conflicts = er.resolve_id_conflicts(tm_match, "tm_player_id", weight_col="minutes")

    overrides = er.load_player_overrides(cfg.paths.external / "manual_player_overrides.csv")
    if not overrides.empty:
        override_map = dict(zip(overrides["fbref_player_id"], overrides["tm_player_id"]))
        hit = tm_match["fbref_player_id"].isin(override_map)
        tm_match.loc[hit, "tm_player_id"] = tm_match.loc[hit, "fbref_player_id"].map(override_map)
        tm_match.loc[hit, "match_method"] = "manual"
        log.info("overrides manuais aplicados", extra={"rows": int(hit.sum())})

    tm_match = _realign_right_index(tm_match, right)

    # --- Understat e Sofascore (estatística: minutagem como validador)
    def _stats(source: str, df: pd.DataFrame, name: str, team: str, rid: str, minutes: str) -> pd.DataFrame:
        left_s = fb.assign(_team=_apply_team_map(fb, maps[source]),
                           _season_minutes=fb.groupby(["season", "fbref_player_id"])["fb_minutes"].transform("sum"))
        out = er.match_players(left_s, df.reset_index(drop=True), left_name="fb_player", right_name=name,
                               left_team="_team", right_team=team, right_id=rid,
                               left_minutes="fb_minutes", right_minutes=minutes,
                               left_minutes_total="_season_minutes", params=params,
                               share_season_aggregate=True)
        out["fbref_player_id"] = fb.loc[out["left_index"], "fbref_player_id"].to_numpy()
        # linha direita compartilhada por mais de um clube FBref = total da temporada
        shared = out["right_index"].notna() & out.duplicated("right_index", keep=False)
        out["is_season_aggregate"] = shared
        return out

    us_match = _stats("understat", us, "us_player", "us_team", "understat_player_id", "us_minutes")
    ss_match = _stats("sofascore", ss, "ss_player", "ss_team", "sofascore_player_id", "ss_minutes_played")

    save_table(team_maps, interim(cfg, "team_maps"), source="entity_resolution.map_teams")
    save_table(tm_match, interim(cfg, "match_fbref_tm"), source="entity_resolution.match_players")
    save_table(us_match, interim(cfg, "match_fbref_understat"), source="entity_resolution.match_players")
    save_table(ss_match, interim(cfg, "match_fbref_sofascore"), source="entity_resolution.match_players")
    return MatchOutputs(team_maps, tm_match, us_match, ss_match, conflicts)


def _realign_right_index(match: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Após conflitos/overrides, aponta `right_index` para uma linha de elenco do id final.

    Preferência: mesma liga-temporada do FBref; senão qualquer linha do jogador (atributos
    estáticos continuam válidos); senão NaN.
    """
    out = match.copy()
    current = out["right_index"].map(lambda ri: right.at[int(ri), "tm_player_id"] if pd.notna(ri) else pd.NA)
    stale = out["tm_player_id"].notna() & (current.astype("Int64") != out["tm_player_id"].astype("Int64")).fillna(True)
    if not stale.any():
        return out
    by_id = right.reset_index().groupby("tm_player_id")
    for i in out.index[stale]:
        pid = int(out.at[i, "tm_player_id"])
        if pid not in by_id.groups:
            out.at[i, "right_index"] = float("nan")
            continue
        rows = by_id.get_group(pid)
        season = out.at[i, "season"] if "season" in out.columns else None
        same = rows[rows["season"] == season] if season is not None else rows.iloc[0:0]
        out.at[i, "right_index"] = float((same if not same.empty else rows)["index"].iloc[0])
    log.info("right_index realinhado após conflitos/overrides", extra={"rows": int(stale.sum())})
    return out


def match_report(fb: pd.DataFrame, match: pd.DataFrame, id_col: str, min_minutes: float = 90) -> pd.DataFrame:
    """Taxa de match por liga-temporada, total e no universo-alvo (linha, ≥ `min_minutes`)."""
    df = fb.reset_index(drop=True).loc[match["left_index"]].assign(
        matched=match[id_col].notna().to_numpy(), method=match["match_method"].to_numpy())
    target = df[(~df["is_goalkeeper"]) & (df["fb_minutes"].fillna(0) >= min_minutes)]
    report = df.groupby(["league", "season"]).agg(rows=("matched", "size"), match_rate=("matched", "mean"))
    report["target_rows"] = target.groupby(["league", "season"]).size()
    report["target_match_rate"] = target.groupby(["league", "season"])["matched"].mean()
    return report.reset_index()


# ----------------------------------------------------------------------------------------------
# 6. Transfermarkt — dados por jogador
# ----------------------------------------------------------------------------------------------
def players_to_fetch(cfg: ExtractionConfig, tm_match: pd.DataFrame, squads: pd.DataFrame) -> pd.DataFrame:
    """Jogadores do Transfermarkt a consultar: só os casados em temporadas de referência.

    As temporadas só de defasagem precisam apenas do desempenho (FBref, Understat, Sofascore); os
    valores passados dos atletas de referência vêm do histórico completo de cada um.
    """
    in_scope = tm_match["season"].isin(cfg.feature_seasons) if "season" in tm_match else True
    ids = tm_match.loc[in_scope, "tm_player_id"].dropna().astype("int64").unique()
    players = (squads[squads["tm_player_id"].isin(ids)]
               .drop_duplicates("tm_player_id")[["tm_player_id", "tm_player_slug", "tm_player"]]
               .sort_values("tm_player_id").reset_index(drop=True))
    missing_slug = set(ids) - set(players["tm_player_id"])
    if missing_slug:
        log.warning("ids casados sem slug de elenco (override manual?)", extra={"n": len(missing_slug)})
        players = pd.concat([players, pd.DataFrame({"tm_player_id": sorted(missing_slug),
                                                    "tm_player_slug": "-", "tm_player": None})],
                            ignore_index=True)
    if cfg.smoke_test:
        players = players.head(cfg.smoke_max_players)
    return players


def run_tm_players(cfg: ExtractionConfig, players: pd.DataFrame) -> dict[str, pd.DataFrame]:
    client = tm_client(cfg)
    failed = cfg.paths.transfermarkt / "failed_fetches.json"
    limit = cfg.max_fetches_per_run

    values, _ = transfermarkt.fetch_for_players(
        client, players, "market_values", lambda pid, _slug: client.market_values(pid), failed, limit)
    profiles, _ = transfermarkt.fetch_for_players(
        client, players, "profiles", lambda pid, slug: client.profile(pid, slug), failed, limit)
    transfers, _ = transfermarkt.fetch_for_players(
        client, players, "transfers", lambda pid, _slug: client.transfers(pid), failed, limit)

    mv_long = pd.DataFrame([{"tm_player_id": pid, **row} for pid, rows in values.items() for row in rows])
    profiles_df = pd.DataFrame([{k: v for k, v in p.items() if k != "raw_fields"} for p in profiles.values()])
    transfers_long = pd.DataFrame([{"tm_player_id": pid, **row} for pid, rows in transfers.items() for row in rows])

    # Quem tem cada recurso coletado: distingue "sem transferências" de "transferências não coletadas"
    status = pd.DataFrame({"tm_player_id": players["tm_player_id"].astype("int64")})
    for kind, got in (("market_values", values), ("profiles", profiles), ("transfers", transfers)):
        status[f"has_{kind}"] = status["tm_player_id"].isin(list(got))
    save_table(mv_long, interim(cfg, "tm_market_values_long"), source="transfermarkt ceapi/marketValueDevelopment")
    save_table(profiles_df, interim(cfg, "tm_profiles"), source="transfermarkt /profil/ (contrato ATUAL)")
    save_table(transfers_long, interim(cfg, "tm_transfers_long"), source="transfermarkt ceapi/transferHistory")
    save_table(status, interim(cfg, "tm_fetch_status"), source="cobertura da coleta por jogador no Transfermarkt")
    return {"market_values": mv_long, "profiles": profiles_df, "transfers": transfers_long, "status": status}


# ----------------------------------------------------------------------------------------------
# 7. Contexto do clube em T (classificação, promoção/rebaixamento, Europa, ClubElo)
# ----------------------------------------------------------------------------------------------
def _sofascore_team_ids(fb: pd.DataFrame, ss: pd.DataFrame, matches: MatchOutputs) -> pd.DataFrame:
    """Id do clube no Sofascore de cada (season, fbref_team_id): a moda entre os atletas pareados."""
    m = matches.sofascore.dropna(subset=["right_index"])
    fb_rows = fb.reset_index(drop=True).loc[m["left_index"].astype(int)]
    ss_rows = ss.reset_index(drop=True).loc[m["right_index"].astype(int)]
    link = pd.DataFrame({"season": fb_rows["season"].to_numpy(), "fbref_team_id": fb_rows["fbref_team_id"].to_numpy(),
                         "sofascore_team_id": ss_rows["sofascore_team_id"].to_numpy()})
    return (link.groupby(["season", "fbref_team_id"])["sofascore_team_id"]
            .agg(lambda v: v.mode().iloc[0]).reset_index())


def run_club_context(cfg: ExtractionConfig, fb: pd.DataFrame, ss: pd.DataFrame, matches: MatchOutputs) -> pd.DataFrame:
    """Contexto de cada clube-temporada de referência; ver `club_context` para fontes e não antecipação."""
    import soccerdata as sd

    seasons = [Season(c) for c in sorted(cfg.feature_seasons)]
    after = seasons[-1].offset(1)
    fb_teams = (fb[["league", "season", "fb_team", "fbref_team_id"]].drop_duplicates()
                .rename(columns={"fb_team": "team"}))
    us_map = matches.team_maps[matches.team_maps["source"] == "understat"]

    # classificação a partir dos resultados (Understat), inclusive a temporada seguinte à última
    reader = sd.Understat(leagues=list(cfg.league_keys), seasons=[s.code for s in seasons] + [after.code],
                          data_dir=cfg.paths.soccerdata_dir / "Understat")
    schedule = reader.read_schedule().reset_index()
    schedule["season"] = schedule["season"].astype(str)
    table = cc.standings(schedule[schedule["season"].isin(cfg.feature_seasons)])

    # pertencimento às Big Five: FBref nas temporadas coletadas; Understat na seguinte à última
    member = fb[["season", "fbref_team_id"]].drop_duplicates().rename(columns={"fbref_team_id": "club"})
    after_teams = set(schedule.loc[schedule["season"] == after.code, ["home_team", "away_team"]].stack())
    last = fb_teams[fb_teams["season"] == seasons[-1].code].merge(us_map, on=["league", "season", "team"])
    if after_teams:
        member = pd.concat([member, pd.DataFrame({"season": after.code,
                                                  "club": last.loc[last["target_team"].isin(after_teams), "fbref_team_id"]})])
    membership = cc.membership_flags(member, [s.code for s in seasons])

    # participação europeia em T (Sofascore, pelo id do clube)
    collector = sofascore.SofascoreCollector(cfg)
    rows = []
    try:
        for s in seasons:
            for _, tid, level in cc.tournaments_for(s.code):
                sid = collector.tournament_season_id(tid, s)
                rows += [{"season": s.code, "sofascore_team_id": t["id"], "level": level}
                         for t in collector.tournament_teams(tid, sid)]
    finally:
        collector.close()
    ids = _sofascore_team_ids(fb, ss, matches)
    participants = pd.DataFrame(rows).merge(ids, on=["season", "sofascore_team_id"]).rename(columns={"fbref_team_id": "club"})
    europe = cc.europe_levels(participants)

    elo = cc.read_clubelo([s.code for s in seasons], sd.ClubElo().read_by_date)
    target = fb_teams[fb_teams["season"].isin(cfg.feature_seasons)]
    elo_map = cc.map_clubelo(target, elo, er.load_aliases(cfg.paths.external / "team_aliases.csv", "clubelo"))
    context = cc.build_club_context(target, us_map, table, membership, europe, elo, elo_map)

    save_table(context, interim(cfg, "club_context"),
               source="Understat (resultados), FBref (pertencimento às ligas), Sofascore (competições UEFA), ClubElo")
    save_table(table, interim(cfg, "club_context_standings"), source="classificação calculada dos resultados do Understat")
    return context


# ----------------------------------------------------------------------------------------------
# 8. Fusão
# ----------------------------------------------------------------------------------------------
TM_STATIC = ["tm_player", "tm_player_slug", "tm_birth_date", "tm_height_cm", "tm_foot", "tm_nationalities"]
TM_SEASONAL = ["tm_team", "tm_club_id", "tm_position", "tm_joined_date_raw", "tm_signed_from", "tm_squad_value_eur"]
PROFILE_COLS = ["current_club", "joined_current_club", "contract_expires", "last_contract_extension",
                "on_loan_from", "parent_contract_expires", "contract_option", "citizenship", "position",
                "height_cm", "scraped_at"]


def build_dataset(
    cfg: ExtractionConfig,
    fb: pd.DataFrame,
    squads: pd.DataFrame,
    us: pd.DataFrame,
    ss: pd.DataFrame,
    matches: MatchOutputs,
    tm_players: dict[str, pd.DataFrame],
    club_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Dataset bruto unificado: uma linha por (temporada, jogador FBref, clube FBref)."""
    base = fb.reset_index(drop=True).copy()
    base["is_lag_only"] = base["season"].isin(cfg.lag_only_seasons)

    # Transfermarkt: id, atributos estáticos, atributos da temporada (só se a linha de elenco é da
    # mesma temporada) e perfil atual.
    tm = matches.tm.set_index("left_index")
    base["tm_player_id"] = tm["tm_player_id"].reindex(base.index).astype("Int64")
    base["tm_match_method"] = tm["match_method"].reindex(base.index)
    base["tm_match_score"] = tm["match_score"].reindex(base.index)
    base["tm_id_conflict"] = tm["id_conflict"].reindex(base.index).fillna(False).astype(bool)

    sq = squads.reset_index(drop=True)
    ref = tm["right_index"].reindex(base.index)
    has_ref = ref.notna()
    ref_rows = sq.loc[ref[has_ref].astype(int)].set_index(base.index[has_ref])
    static = sq.drop_duplicates("tm_player_id").set_index("tm_player_id")[TM_STATIC]
    base = base.join(static, on="tm_player_id")
    same_season = has_ref & (ref_rows.reindex(base.index)["season"] == base["season"])
    for col in TM_SEASONAL:
        base[col] = ref_rows[col].reindex(base.index).where(same_season)

    profiles = tm_players["profiles"]
    if not profiles.empty:
        prof = profiles.set_index("tm_player_id")[[c for c in PROFILE_COLS if c in profiles.columns]]
        base = base.join(prof.add_prefix("tm_profile_"), on="tm_player_id")

    # Valores de mercado em T, T+1, T+2
    if not tm_players["market_values"].empty:
        base = attach_values(base, tm_players["market_values"], horizons=(0, 1, 2),
                             max_staleness_days=cfg.value_max_staleness_days)

    # Understat / Sofascore: linha exata casada
    for source, frame, match, keep_prefix, ids in (
        ("understat", us, matches.understat, "us_", ["understat_player_id", "understat_team_id"]),
        ("sofascore", ss, matches.sofascore, "ss_", ["sofascore_player_id", "sofascore_team_id"]),
    ):
        cols = ids + [c for c in frame.columns if c.startswith(keep_prefix)]
        m = match.set_index("left_index")
        rows = frame.reset_index(drop=True).loc[m["right_index"].dropna().astype(int), cols]
        rows.index = m.index[m["right_index"].notna()]
        base = base.join(rows)
        base[f"{keep_prefix}match_method"] = m["match_method"].reindex(base.index)
        base[f"{keep_prefix}match_score"] = m["match_score"].reindex(base.index)
        base[f"{keep_prefix}is_season_aggregate"] = m["is_season_aggregate"].reindex(base.index).fillna(False).astype(bool)

    # Contexto do clube na temporada (prefixo ctx_): uma linha por clube-temporada de referência
    if club_context is not None and not club_context.empty:
        ctx = club_context[["season", "fbref_team_id", "league_rank_T", "league_points_T", *cc.CONTEXT_FEATURES]]
        base = base.merge(ctx.rename(columns={c: f"ctx_{c}" for c in ctx.columns if c not in ("season", "fbref_team_id")}),
                          on=["season", "fbref_team_id"], how="left")

    key = ["season", "fbref_player_id", "fbref_team_id"]
    if base.duplicated(key).any():
        raise ValueError("chave (season, fbref_player_id, fbref_team_id) duplicada após a fusão")

    lead = ["season", "league", "is_lag_only", "fbref_player_id", "fb_player", "fbref_team_id", "fb_team",
            "fb_position", "is_goalkeeper", "fb_birth_year", "fb_age", "fb_nationality", "n_teams_in_season",
            "tm_player_id", "tm_match_method", "tm_match_score"]
    base = base[lead + [c for c in base.columns if c not in lead]]
    return base.sort_values(["season", "league", "fb_team", "fb_player"]).reset_index(drop=True)


def column_dictionary(df: pd.DataFrame) -> dict[str, str]:
    source = {"fb_": "FBref", "us_": "Understat", "ss_": "Sofascore", "ctx_": "Contexto do clube em T (Transfermarkt + ClubElo)",
              "tm_profile_": "Transfermarkt (perfil, contrato ATUAL)",
              "tm_value_": "Transfermarkt (histórico de valores; último valor ≤ 30/jun de T+k)", "tm_": "Transfermarkt (elenco)"}
    out = {}
    for col in df.columns:
        prefix = next((p for p in source if col.startswith(p)), None)
        out[col] = source[prefix] if prefix else "identificação / controle"
    return out


def save_dataset(cfg: ExtractionConfig, df: pd.DataFrame) -> None:
    save_table(df, cfg.paths.final_dataset, source="notebook 01 — FBref + Understat + Sofascore + Transfermarkt",
               description=column_dictionary(df))
    df.to_csv(cfg.paths.final_dataset.with_suffix(".csv"), index=False)
    log.info("dataset final gravado", extra={"path": str(cfg.paths.final_dataset), "rows": len(df)})


def coverage(df: pd.DataFrame, min_minutes: float = 90) -> pd.DataFrame:
    """Cobertura por liga-temporada no universo-alvo (jogadores de linha com ≥ `min_minutes`)."""
    target = df[(~df["is_goalkeeper"]) & (df["fb_minutes"].fillna(0) >= min_minutes) & (~df["is_lag_only"])]
    g = target.groupby(["league", "season"])
    out = pd.DataFrame({
        "rows": g.size(),
        "tm_matched": g["tm_player_id"].apply(lambda s: s.notna().mean()),
        "understat_matched": g["understat_player_id"].apply(lambda s: s.notna().mean()),
        "sofascore_matched": g["sofascore_player_id"].apply(lambda s: s.notna().mean()),
    })
    for k in ("T", "T1", "T2"):
        col = f"tm_value_{k}"
        if col in target.columns:
            out[f"value_{k}"] = g[col].apply(lambda s: s.notna().mean())
    return out.reset_index()


def load_interim(cfg: ExtractionConfig, name: str) -> pd.DataFrame:
    return load_table(interim(cfg, name))
