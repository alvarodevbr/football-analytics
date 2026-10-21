import numpy as np
import pandas as pd

from football_analytics import entity_resolution as er


def _fb(rows):
    return pd.DataFrame(rows, columns=["league", "season", "fb_team", "fbref_player_id", "fb_player",
                                       "fb_birth_year", "fb_minutes"])


def _tm(rows):
    return pd.DataFrame(rows, columns=["league", "season", "tm_team", "tm_player_id", "tm_player", "tm_birth_year"])


def _match_tm(fb, tm, **kw):
    return er.match_players(fb, tm, left_name="fb_player", right_name="tm_player", left_team="fb_team",
                            right_team="tm_team", right_id="tm_player_id", left_birth="fb_birth_year",
                            right_birth="tm_birth_year", **kw)


def test_homonyms_resolved_by_birth_year():
    fb = _fb([("L", "2324", "X", "f1", "Danilo", 1991, 2000),
              ("L", "2324", "X", "f2", "Danilo", 2001, 300)])
    tm = _tm([("L", "2324", "X", 10, "Danilo", 2001),
              ("L", "2324", "X", 11, "Danilo", 1991)])
    out = _match_tm(fb, tm)
    assert list(out["tm_player_id"]) == [11, 10]


def test_one_to_one_and_birth_gap_blocks_false_positive():
    fb = _fb([("L", "2324", "X", "f1", "Bernardo Silva", 1994, 2500),
              ("L", "2324", "X", "f2", "Andre Silva", 1995, 900)])
    tm = _tm([("L", "2324", "X", 20, "Bernardo Silva", 1994),
              ("L", "2324", "X", 21, "André Silva", 1980)])  # ano de nascimento incompatível
    out = _match_tm(fb, tm)
    assert out.loc[0, "tm_player_id"] == 20
    assert np.isnan(out.loc[1, "tm_player_id"]) and out.loc[1, "match_method"] == "unmatched"


def test_accents_and_nicknames():
    fb = _fb([("L", "2324", "X", "f1", "Martin Ødegaard", 1998, 3000),
              ("L", "2324", "X", "f2", "Gabriel Magalhães", 1997, 3000),
              ("L", "2324", "X", "f3", "Rodri", 1996, 3000)])
    tm = _tm([("L", "2324", "X", 30, "Martin Odegaard", 1998),
              ("L", "2324", "X", 31, "Gabriel", 1997),
              ("L", "2324", "X", 32, "Rodrigo Hernández", 1996)])
    out = _match_tm(fb, tm)
    assert list(out["tm_player_id"]) == [30, 31, 32]
    assert out.loc[2, "match_method"] == "residual"


def test_league_fallback_for_wrong_club_block():
    # emprestado no meio da temporada: o FBref lista o clube Y, o elenco TM só o tem no clube Z
    fb = _fb([("L", "2324", "Y", "f1", "Jhon Duran", 2003, 400)])
    tm = _tm([("L", "2324", "Y", 40, "Someone Else", 1990),
              ("L", "2324", "Z", 41, "Jhon Durán", 2003)])
    out = _match_tm(fb, tm)
    assert out.loc[0, "tm_player_id"] == 41 and out.loc[0, "match_method"] == "league_fallback"


def test_propagation_by_fbref_id_across_seasons():
    fb = _fb([("L", "2223", "X", "f1", "Vitinha", 2000, 2000),
              ("L", "2324", "X", "f1", "Vítor Machado Ferreira", 2000, 2000)])
    tm = _tm([("L", "2223", "X", 50, "Vitinha", 2000),
              ("L", "2324", "X", 50, "Vitinha", 2000),
              ("L", "2324", "X", 51, "Vitor Roque", 2005)])
    out = _match_tm(fb, tm, cross_season=True)
    assert list(out["tm_player_id"]) == [50, 50]
    assert out.loc[1, "match_method"] == "propagated"


def test_stats_source_uses_minutes_as_validator():
    fb = _fb([("L", "2324", "X", "f1", "Kieran Tierney", 1997, 1500)])
    us = pd.DataFrame({"league": ["L", "L"], "season": ["2324", "2324"], "us_team": ["X", "X"],
                       "understat_player_id": [1, 2], "us_player": ["Kieran Tierney", "Kieran Trippier"],
                       "us_minutes": [1490, 30]})
    out = er.match_players(fb, us, left_name="fb_player", right_name="us_player", left_team="fb_team",
                           right_team="us_team", right_id="understat_player_id",
                           left_minutes="fb_minutes", right_minutes="us_minutes")
    assert out.loc[0, "understat_player_id"] == 1


def test_resolve_id_conflicts_majority_by_minutes():
    df = pd.DataFrame({"left_index": [0, 1, 2], "fbref_player_id": ["f1", "f1", "f1"],
                       "tm_player_id": [7.0, 7.0, 8.0], "minutes": [2000, 1500, 100],
                       "match_method": ["exact", "exact", "fuzzy"]})
    out, report = er.resolve_id_conflicts(df, "tm_player_id", weight_col="minutes")
    assert list(out["tm_player_id"]) == [7.0, 7.0, 7.0]
    assert out.loc[2, "match_method"] == "conflict_resolved"
    assert out["id_conflict"].all() and len(report) == 2


def test_map_teams_assignment_and_alias():
    left = pd.DataFrame({"league": "L", "season": "2324",
                         "team": ["Manchester Utd", "Manchester City", "Wolves", "Nott'ham Forest"]})
    right = pd.DataFrame({"league": "L", "season": "2324",
                          "team": ["Manchester City", "Manchester United", "Wolverhampton Wanderers",
                                   "Nottingham Forest"]})
    out = er.map_teams(left, right, aliases={"Wolves": "Wolverhampton Wanderers"}).set_index("team")
    assert out.loc["Manchester Utd", "target_team"] == "Manchester United"
    assert out.loc["Manchester City", "target_team"] == "Manchester City"
    assert out.loc["Wolves", "method"] == "alias"
    assert out.loc["Nott'ham Forest", "target_team"] == "Nottingham Forest"


def test_diminutive_first_names_match_by_surname_and_initial():
    fb = _fb([("L", "2324", "X", "f1", "Max Kilman", 1997, 3400),
              ("L", "2324", "X", "f2", "Jáder Durán", 2003, 400),
              ("L", "2324", "X", "f3", "Tom Smith", 1999, 900)])
    tm = _tm([("L", "2324", "X", 60, "Maximilian Kilman", 1997),
              ("L", "2324", "X", 61, "Jhon Durán", 2003),
              ("L", "2324", "X", 62, "Tom Bloggs", 1999),
              ("L", "2324", "X", 63, "Youth Player", 2006)])  # 2 sobras: a regra residual não se aplica
    out = _match_tm(fb, tm)
    assert out.loc[0, "tm_player_id"] == 60 and out.loc[1, "tm_player_id"] == 61
    assert np.isnan(out.loc[2, "tm_player_id"])  # mesma inicial sem o mesmo sobrenome não basta


def test_season_aggregate_row_shared_by_both_clubs():
    # transferido em janeiro: FBref tem duas linhas; Understat, uma com o total e os dois clubes
    fb = _fb([("L", "2324", "City", "f1", "Cole Palmer", 2002, 11),
              ("L", "2324", "Chelsea", "f1", "Cole Palmer", 2002, 2617)])
    fb["season_minutes"] = fb.groupby(["season", "fbref_player_id"])["fb_minutes"].transform("sum")
    us = pd.DataFrame({"league": ["L"], "season": ["2324"], "us_team": ["Manchester City,Chelsea"],
                       "understat_player_id": [9], "us_player": ["Cole Palmer"], "us_minutes": [2628]})
    out = er.match_players(fb, us, left_name="fb_player", right_name="us_player", left_team="fb_team",
                           right_team="us_team", right_id="understat_player_id", left_minutes="fb_minutes",
                           right_minutes="us_minutes", left_minutes_total="season_minutes",
                           share_season_aggregate=True)
    assert list(out["understat_player_id"]) == [9, 9]
    assert set(out["match_method"]) == {"league_fallback", "season_aggregate"}
