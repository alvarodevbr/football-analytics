import numpy as np
import pandas as pd

from football_analytics import contracts as ct
from football_analytics import features as ft
from football_analytics.validation import expanding_window_folds

A = pd.Timestamp("2021-06-30")


def _moves(rows):
    df = pd.DataFrame(rows, columns=["transfer_date", "from_club_id", "to_club_id", "to_club", "fee_text",
                                     "is_loan", "is_end_of_loan"])
    df["transfer_date"] = pd.to_datetime(df["transfer_date"])
    return df


def _profile(**kw):
    base = {"contract_expires": None, "last_contract_extension": None, "on_loan_from": None,
            "parent_contract_expires": None}
    return pd.Series({**base, **kw})


def test_current_contract_observed_when_signed_before_anchor():
    moves = _moves([("2019-07-01", 1, 10, "Club X", "€5.00m", False, False)])
    c = ct.contract_for(A, moves, _profile(contract_expires="2024-06-30"))
    assert c.method == "current_contract" and c.expiry == pd.Timestamp("2024-06-30")


def test_extension_after_anchor_is_estimated():
    moves = _moves([("2019-07-01", 1, 10, "Club X", "€5.00m", False, False)])
    c = ct.contract_for(A, moves, _profile(contract_expires="2027-06-30", last_contract_extension="2022-03-01"))
    assert c.method == "extension_after_T" and c.expiry is None


def test_contract_longer_than_five_years_from_signing_implies_unrecorded_extension():
    moves = _moves([("2015-07-01", 1, 10, "Club X", "€5.00m", False, False)])
    c = ct.contract_for(A, moves, _profile(contract_expires="2025-06-30"))
    assert c.method == "unrecorded_extension"


def test_free_exit_gives_expiry_and_paid_exit_is_estimated():
    free = _moves([("2019-07-01", 1, 10, "Club X", "€5.00m", False, False),
                   ("2022-07-01", 10, 20, "Club Y", "free transfer", False, False)])
    c = ct.contract_for(A, free, None)
    assert c.method == "free_exit" and c.expiry == pd.Timestamp("2022-06-30")
    paid = free.assign(fee_text=["€5.00m", "€12.00m"])
    assert ct.contract_for(A, paid, None).method == "transfer_exit"


def test_loan_out_does_not_end_contract_and_loan_at_anchor_is_flagged():
    moves = _moves([("2019-07-01", 1, 10, "Club X", "€5.00m", False, False),
                    ("2021-08-01", 10, 30, "Club Z", "loan transfer", True, False),
                    ("2022-06-30", 30, 10, "Club X", "End of loan", False, True),
                    ("2023-07-01", 10, 515, "Without Club", "-", False, False)])
    c = ct.contract_for(A, moves, None)
    assert c.method == "free_exit" and c.expiry == pd.Timestamp("2023-07-01") - pd.Timedelta(days=1)
    on_loan = ct.contract_for(pd.Timestamp("2022-01-31"), moves, None)
    assert on_loan.method == "on_loan" and on_loan.on_loan


def test_contracts_at_anchor_estimates_and_bosman():
    rows = pd.DataFrame({"tm_player_id": [1, 2], "anchor_date_T": [A, A], "age_band": ["24–28", "24–28"]})
    tr = pd.DataFrame({
        "tm_player_id": [1, 2, 2], "transfer_date": ["2020-07-01", "2020-07-01", "2022-01-15"],
        "from_club_id": [5, 5, 10], "to_club_id": [10, 10, 40], "to_club": ["X", "X", "W"],
        "fee_text": ["€1.00m", "€1.00m", "€3.00m"], "is_loan": [False] * 3, "is_end_of_loan": [False] * 3,
        "upcoming": [False] * 3})
    prof = pd.DataFrame({"tm_player_id": [1], "contract_expires": ["2022-06-30"], "last_contract_extension": [None],
                         "on_loan_from": [None], "parent_contract_expires": [None]})
    out = ct.contracts_at_anchor(rows, tr, prof)
    assert out.loc[0, "contract_method"] == "current_contract" and out.loc[0, "bosman_T"] == 1.0
    assert out.loc[1, "contract_is_estimated"]
    assert np.isclose(out.loc[1, "contract_years_remaining_T"], out.loc[0, "contract_years_remaining_T"])
    assert np.isclose(out.loc[0, "tenure_years_T"], 1.0, atol=0.01)


def test_nationality_features():
    df = pd.DataFrame({"tm_nationalities": ["Brazil|Portugal", "England", "England", "Senegal", None],
                       "season": ["2223", "1920", "2223", "2223", "2223"],
                       "league": ["ESP-La Liga", "ENG-Premier League", "ITA-Serie A", "FRA-Ligue 1", "ESP-La Liga"]})
    out = ft.nationality_features(df)
    assert list(out["confederation"][:4]) == ["CONMEBOL", "UEFA", "UEFA", "CAF"]
    assert list(out["has_eu_passport"][:4]) == [1.0, 1.0, 0.0, 0.0]  # Brexit: britânico deixa de ser comunitário
    assert out.loc[1, "is_domestic"] == 1.0 and out.loc[2, "is_domestic"] == 0.0
    assert np.isnan(out.loc[4, "is_uefa"])


def test_aggregate_source_counts_shared_season_row_once():
    raw = pd.DataFrame({
        "season": ["2324"] * 3, "fbref_player_id": ["p", "p", "q"],
        "understat_player_id": [9, 9, 7], "understat_team_id": ["A,B", "A,B", "C"],
        "us_minutes": [2000.0, 2000.0, 900.0], "us_xg": [5.0, 5.0, 1.0]})
    out = ft._aggregate_source(raw, ["understat_player_id", "understat_team_id"], ("us_minutes", "us_xg"),
                               "us_minutes").set_index("fbref_player_id")
    assert out.loc["p", "us_xg"] == 5.0 and out.loc["p", "us_minutes"] == 2000.0


def test_expanding_window_folds_with_purge():
    seasons = ["1920", "2021", "2122", "2223", "2324"]
    f1 = expanding_window_folds(seasons, horizon=1)
    assert [(f.train[-1], f.test) for f in f1] == [("1920", "2021"), ("2021", "2122"), ("2122", "2223"),
                                                  ("2223", "2324")]
    f2 = expanding_window_folds(seasons, horizon=2)
    assert [(f.train, f.purged, f.test) for f in f2][0] == (("1920",), ("2021",), "2122")
    assert len(f2) == 3 and all(len(f.purged) == 1 for f in f2)
