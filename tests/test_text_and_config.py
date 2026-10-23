from datetime import date

import pytest

from football_analytics.config import Season, default_config
from football_analytics.text import normalize_club, normalize_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Martin Ødegaard", "martin odegaard"),
        ("N'Golo Kanté", "ngolo kante"),
        ("Jean-Philippe Mateta", "jean philippe mateta"),
        ("Kamil Glik", "kamil glik"),
        ("Łukasz Skorupski", "lukasz skorupski"),
        ("Çağlar Söyüncü", "caglar soyuncu"),
        ("Thomas Müller", "thomas muller"),
        ("Matthijs de Ligt ", "matthijs de ligt"),
        ("Strahinja Pavlović", "strahinja pavlovic"),
        ("Joško Gvardiol", "josko gvardiol"),
        (None, ""),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_club_keeps_city_rivals_apart():
    assert normalize_club("Manchester United") != normalize_club("Manchester City")
    assert normalize_club("AC Milan") == normalize_club("Milan")


def test_season_conversions():
    s = Season("2324")
    assert s.start_year == 2023
    assert s.fbref_label == "2023-2024"
    assert s.tm_saison_id == "2023"
    assert s.sofascore_year == "23/24"
    assert s.end_anchor == date(2024, 6, 30)
    assert s.offset(2).code == "2526"
    assert Season("1920").offset(-1).code == "1819"


@pytest.mark.parametrize("bad", ["2325", "abcd", "23"])
def test_season_rejects_invalid(bad):
    with pytest.raises(ValueError):
        Season(bad)


def test_smoke_config_isolates_data(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    (tmp_path / "src" / "football_analytics").mkdir(parents=True)
    cfg = default_config(root=tmp_path)
    smoke = cfg.as_smoke_test()
    assert smoke.paths.raw != cfg.paths.raw
    assert smoke.paths.external == cfg.paths.external
    assert [s.code for s in smoke.seasons] == ["2324"]
    assert [s.code for s in cfg.seasons] == ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]
    assert cfg.target_seasons(2)[-1] == "2324" and cfg.target_seasons(1)[-1] == "2425"
