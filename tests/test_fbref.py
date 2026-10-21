import numpy as np
import pandas as pd
import pytest

from football_analytics import fbref


def test_parse_player_table_ids_and_identity(fixture_text):
    df = fbref.parse_player_table(fixture_text("fbref_big5_passing_2223.html"), "passing")
    assert len(df) == 6  # a linha de cabeçalho repetida é ignorada
    assert df["fbref_player_id"].str.fullmatch(r"[0-9a-f]{8}").all()
    assert df["fbref_team_id"].str.fullmatch(r"[0-9a-f]{8}").all()
    assert set(df["comp"]) <= {"Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1"}
    assert df["birth_year"].between(1975, 2010).all()
    assert (df["nationality"].str.len() == 3).all()


def test_opta_columns_are_empty_not_zero(fixture_text):
    df = fbref.parse_player_table(fixture_text("fbref_big5_passing_2223.html"), "passing")
    assert df["passes_completed"].isna().all()
    assert df["assists"].notna().all()
    rates = fbref.column_fill_rates(df)
    assert rates["passes_progressive_distance"] == 0.0


def test_missing_table_raises():
    with pytest.raises(fbref.FBrefParseError):
        fbref.parse_player_table("<html><body><p>Just a moment...</p></body></html>", "standard")


def test_merge_pages_maps_league_and_keeps_base_values():
    std = pd.DataFrame({"fbref_player_id": ["a", "b"], "player": ["A", "B"], "fbref_team_id": ["t1", "t2"],
                        "team": ["X", "Y"], "comp": ["Premier League", "Bundesliga"], "nationality": ["ENG", "GER"],
                        "position": ["DF", "FW"], "age": [20, 25], "birth_year": [2002, 1997],
                        "goals": [1.0, 5.0], "minutes": [900.0, 1800.0]})
    shooting = std[["fbref_player_id", "player", "fbref_team_id", "team", "comp", "nationality", "position",
                    "age", "birth_year", "goals"]].assign(shots=[3.0, np.nan])
    merged = fbref.merge_pages({"standard": std, "shooting": shooting},
                               {"Premier League": "ENG-Premier League", "Bundesliga": "GER-Bundesliga"})
    assert list(merged["league"]) == ["ENG-Premier League", "GER-Bundesliga"]
    assert list(merged["goals"]) == [1.0, 5.0]
    assert merged["shots"].iloc[0] == 3.0 and np.isnan(merged["shots"].iloc[1])


def test_merge_pages_rejects_unknown_competition():
    std = pd.DataFrame({"fbref_player_id": ["a"], "player": ["A"], "fbref_team_id": ["t"], "team": ["X"],
                        "comp": ["Eredivisie"], "nationality": ["NED"], "position": ["MF"], "age": [20],
                        "birth_year": [2002]})
    with pytest.raises(fbref.FBrefParseError):
        fbref.merge_pages({"standard": std}, {"Premier League": "ENG-Premier League"})
