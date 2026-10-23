import numpy as np
import pandas as pd

from football_analytics import club_context as cc


def _schedule() -> pd.DataFrame:
    games = [("A", "B", 2, 0), ("B", "A", 1, 1), ("A", "C", 3, 1), ("C", "A", 0, 0), ("B", "C", 1, 0), ("C", "B", 2, 2)]
    return pd.DataFrame([{"league": "L", "season": "2324", "home_team": h, "away_team": a, "home_goals": hg,
                          "away_goals": ag, "is_result": True} for h, a, hg, ag in games]
                        + [{"league": "L", "season": "2324", "home_team": "A", "away_team": "B", "home_goals": np.nan,
                            "away_goals": np.nan, "is_result": False}])


def test_standings_from_results():
    t = cc.standings(_schedule()).set_index("team")
    assert t.loc["A", "points"] == 8 and t.loc["A", "matches"] == 4 and t.loc["A", "rank"] == 1
    assert t.loc["A", "league_rank_pct_T"] == 0 and t.loc["C", "league_rank_pct_T"] == 1   # C: 2 pontos
    assert np.isclose(t.loc["A", "league_gd_per_match_T"], (6 - 2) / 4)
    assert t["matches"].eq(4).all()                                                         # jogo sem resultado fora


def test_standings_ranks_by_points_per_match_when_matches_differ():
    # liga interrompida (Ligue 1 2019/20): a ordem usa pontos por jogo
    s = pd.DataFrame([{"league": "L", "season": "1920", "home_team": h, "away_team": a, "home_goals": hg,
                       "away_goals": ag, "is_result": True}
                      for h, a, hg, ag in [("X", "Y", 1, 0), ("X", "Z", 1, 0), ("Y", "Z", 1, 0), ("Z", "W", 1, 0),
                                           ("Z", "Y", 1, 0), ("W", "Y", 1, 0)]])
    t = cc.standings(s).set_index("team")
    assert t.loc["X", "rank"] == 1          # 6 pontos em 2 jogos (3,0 por jogo) à frente de Z (6 em 4)


def test_membership_flags_promotion_and_relegation():
    clubs = pd.DataFrame({"season": ["2122"] * 3 + ["2223"] * 3 + ["2324"] * 3,
                          "club": ["a", "b", "c", "a", "b", "d", "a", "d", "e"]})
    out = cc.membership_flags(clubs, ["2223", "2324"]).set_index(["season", "club"])
    assert out.loc[("2223", "d"), "promoted_T"] == 1 and out.loc[("2223", "a"), "promoted_T"] == 0
    assert out.loc[("2223", "b"), "relegated_T"] == 1 and out.loc[("2223", "d"), "relegated_T"] == 0
    assert np.isnan(out.loc[("2324", "e"), "relegated_T"])  # 2024/25 não coletada
    assert out.loc[("2324", "e"), "promoted_T"] == 1


def test_europe_levels_keeps_highest_competition():
    part = pd.DataFrame({"season": ["2324"] * 3, "club": ["x", "x", "y"], "level": [3, 2, 1]})
    out = cc.europe_levels(part).set_index("club")["europe_T"]
    assert out["x"] == 3 and out["y"] == 1


def test_tournaments_for_conference_league_start():
    assert [t[1] for t in cc.tournaments_for("2021")] == [7, 679]
    assert [t[1] for t in cc.tournaments_for("2122")] == [7, 679, 17015]


def test_clubelo_dates_anchor():
    d = cc.clubelo_dates("2324")
    assert d["anchor"].isoformat() == "2024-06-30" and d["start"].isoformat() == "2023-08-01"
    assert d["mid"].isoformat() == "2024-01-15"


def test_build_club_context_joins_all_blocks():
    fb_teams = pd.DataFrame({"league": ["L"] * 3, "season": ["2324"] * 3, "team": ["Alpha", "Beta", "Gamma"],
                             "fbref_team_id": ["fa", "fb", "fc"]})
    us_map = pd.DataFrame({"league": ["L"] * 3, "season": ["2324"] * 3, "team": ["Alpha", "Beta", "Gamma"],
                           "target_team": ["A", "B", "C"]})
    table = cc.standings(_schedule())
    membership = pd.DataFrame({"season": ["2324"] * 3, "club": ["fa", "fb", "fc"], "promoted_T": [0.0, 0.0, 1.0],
                               "relegated_T": [0.0, 0.0, 1.0]})
    europe = pd.DataFrame({"season": ["2324"], "club": ["fa"], "europe_T": [3]})
    elo = pd.DataFrame({"team": ["EA", "EA"], "league": ["L"] * 2, "elo": [1800, 1850], "season": ["2324"] * 2,
                        "kind": ["start", "anchor"]})
    elo_map = pd.DataFrame({"league": ["L"], "season": ["2324"], "team": ["Alpha"], "target_team": ["EA"]})
    out = cc.build_club_context(fb_teams, us_map, table, membership, europe, elo, elo_map).set_index("fbref_team_id")
    assert out.loc["fa", "league_rank_pct_T"] == 0 and out.loc["fc", "league_rank_pct_T"] == 1
    assert out.loc["fa", "europe_T"] == 3 and out.loc["fb", "europe_T"] == 0   # sem Europa = 0
    assert out.loc["fc", "relegated_T"] == 1 and out.loc["fc", "promoted_T"] == 1
    assert out.loc["fa", "club_elo_T"] == 1850 and out.loc["fa", "club_elo_change_T"] == 50
    assert np.isnan(out.loc["fb", "club_elo_T"])
    # sem ClubElo (API fora do ar): o contexto continua, com o Elo ausente
    empty = pd.DataFrame(columns=cc.ELO_COLUMNS)
    no_elo = cc.build_club_context(fb_teams, us_map, table, membership, europe, empty, cc.map_clubelo(fb_teams, empty))
    assert no_elo["club_elo_T"].isna().all() and no_elo["relegated_T"].notna().all()


def test_read_clubelo_degrades_on_connection_error():
    calls = []

    def failing(when: str) -> pd.DataFrame:
        calls.append(when)
        raise ConnectionError("502")

    out = cc.read_clubelo(["2223", "2324"], failing)
    assert out.empty and list(out.columns) == cc.ELO_COLUMNS and len(calls) == 1  # para na sonda
