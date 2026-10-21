import json

import pytest

from football_analytics import transfermarkt as tm
from conftest import FIXTURES


@pytest.mark.parametrize(
    ("text", "expected"),
    [("€35.00m", 35e6), ("€500k", 5e5), ("€1.20bn", 1.2e9), ("-", None), ("", None),
     ("Loan fee: €2.90m", 2.9e6)],
)
def test_parse_value_eur(text, expected):
    assert tm.parse_value_eur(text) == expected


def test_parse_date_and_height():
    assert tm.parse_date("09/09/2001 (20)") == "2001-09-09"
    assert tm.parse_date("31/02/2020") is None
    assert tm.parse_height_cm("1,98m") == 198.0
    assert tm.parse_height_cm("1,78\xa0m") == 178.0


def test_parse_squad(fixture_text):
    rows = tm.parse_squad(fixture_text("tm_squad_arsenal_2021.html"))
    assert len(rows) == 6
    okonkwo = next(r for r in rows if r["tm_player_id"] == 503769)
    assert okonkwo["tm_player"] == "Arthur Okonkwo"
    assert okonkwo["tm_player_slug"] == "arthur-okonkwo"
    assert okonkwo["tm_position"] == "Goalkeeper"
    assert okonkwo["tm_birth_date"] == "2001-09-09"
    assert okonkwo["tm_nationalities"] == "Nigeria|England"
    assert okonkwo["tm_height_cm"] == 198.0
    assert okonkwo["tm_foot"] == "right"
    assert okonkwo["tm_joined_date_raw"] == "2021-07-01"
    assert okonkwo["tm_signed_from"] == "Arsenal FC U23"
    assert okonkwo["tm_squad_value_eur"] == 1.5e6
    # jogador da base sem data de chegada nem valor: campos vazios, não deslocados
    kirk = next(r for r in rows if r["tm_player"] == "Alex Kirk")
    assert kirk["tm_birth_date"] == "2002-10-27"
    assert kirk["tm_joined_date_raw"] is None
    assert kirk["tm_squad_value_eur"] is None
    assert kirk["tm_height_cm"] == 190.0


def test_parse_profile_contract_fields(fixture_text):
    saka = tm.parse_profile(fixture_text("tm_profile_saka.html"))
    assert saka["contract_expires"] == "2030-06-30"
    assert saka["joined_current_club"] == "2019-07-01"
    assert saka["last_contract_extension"] == "2026-02-19"
    assert saka["current_club"] == "Arsenal FC"
    assert saka["citizenship"] == "England | Nigeria"
    assert saka["height_cm"] == 178.0
    felix = tm.parse_profile(fixture_text("tm_profile_loan.html"))
    assert felix["last_contract_extension"] is None
    assert felix["contract_expires"] == "2027-06-30"


def test_parse_market_values():
    payload = json.loads((FIXTURES / "tm_market_values_felix.json").read_text())
    rows = tm.parse_market_values(payload)
    assert rows[0] == {"mv_date": "2016-09-30", "mv_value_eur": 25000.0, "mv_club": "SL Benfica U19", "mv_age": "16"}
    assert all(r["mv_date"] <= s["mv_date"] for r, s in zip(rows, rows[1:]))


def test_parse_transfers_loan_flags():
    payload = json.loads((FIXTURES / "tm_transfers_felix.json").read_text())
    rows = tm.parse_transfers(payload)
    by_date = {r["transfer_date"]: r for r in rows}
    assert by_date["2025-07-29"]["fee_eur"] == 35e6 and not by_date["2025-07-29"]["is_loan"]
    assert by_date["2025-02-03"]["is_loan"] and by_date["2025-02-03"]["fee_eur"] == 2.9e6
    assert by_date["2025-06-30"]["is_end_of_loan"]
    assert by_date["2019-07-03"]["from_club"] == "Benfica"
    assert by_date["2019-07-03"]["to_club_id"] is not None


def test_http_get_retries_5xx_then_succeeds(monkeypatch):
    import requests

    from football_analytics import http

    calls = {"n": 0}

    class FakeResp:
        def __init__(self, status):
            self.status_code = status
            self.headers = {}

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.RetryError("too many 502 error responses")
            return FakeResp(502 if calls["n"] == 2 else 200)

    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    resp = http.get(FakeSession(), "https://x", http.Throttle(0), attempts=4)
    assert resp.status_code == 200 and calls["n"] == 3


def test_http_get_persistent_403_raises_blocked(monkeypatch):
    from football_analytics import http

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            return type("R", (), {"status_code": 403, "headers": {}})()

    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(http.BlockedError):
        http.get(FakeSession(), "https://x", http.Throttle(0), attempts=3)


def test_http_get_waf_captcha_stops_immediately(monkeypatch):
    from football_analytics import http

    calls = {"n": 0}

    class FakeSession:
        def get(self, url, headers=None, timeout=None):
            calls["n"] += 1
            return type("R", (), {"status_code": 405, "headers": {"x-amzn-waf-action": "captcha"}})()

    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    with pytest.raises(http.BlockedError):
        http.get(FakeSession(), "https://x", http.Throttle(0), attempts=5)
    assert calls["n"] == 1


def test_fetch_for_players_keeps_reading_cache_after_block(tmp_path):
    import pandas as pd

    from football_analytics.http import BlockedError

    cached = {1, 3}

    class FakeClient:
        network_fetches = 0

        def is_cached(self, kind, pid):
            return pid in cached

    def fetch_one(pid, _slug):
        if pid not in cached:
            raise BlockedError("waf")
        return {"pid": pid}

    players = pd.DataFrame({"tm_player_id": [1, 2, 3, 4], "tm_player_slug": ["a", "b", "c", "d"]})
    results, failures = tm.fetch_for_players(FakeClient(), players, "market_values", fetch_one,
                                             tmp_path / "failed.json")
    assert set(results) == {1, 3}      # o cache depois do bloqueio continua sendo lido
    assert failures == {2}             # e o jogador 4 nem chega a ser requisitado
