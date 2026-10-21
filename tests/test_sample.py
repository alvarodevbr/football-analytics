import numpy as np
import pandas as pd

from football_analytics import sample as sp


def test_position_group_prefers_tm_and_falls_back_to_fbref():
    tm = pd.Series(["Attacking Midfield", None, "Goalkeeper", "Right Winger"])
    fb = pd.Series(["FW", "DF,MF", "GK", "MF"])
    assert list(sp.position_group(tm, fb)) == ["MEI", "DEF", "GK", "ATA"]


def test_completed_years_and_band_boundaries():
    birth = pd.to_datetime(pd.Series(["2000-07-01", "2000-06-30", "1991-06-30", "1990-12-31"]))
    ref = pd.Series(pd.to_datetime(["2024-06-30"] * 4))
    age = sp.completed_years(birth, ref)
    assert list(age) == [23, 24, 33, 33]
    assert list(sp.age_band(age)) == ["16–23", "24–28", "33+", "33+"]
    assert list(sp.age_band(pd.Series([28, 29, 32], dtype="Int64"))) == ["24–28", "29–32", "29–32"]


def test_category_ids_follow_spec_table():
    band = pd.Series(["16–23", "16–23", "24–28", "33+", None])
    pos = pd.Series(["DEF", "ATA", "MEI", "ATA", "DEF"])
    out = sp.category_id(band, pos)
    assert list(out[:4]) == ["C1", "C3", "C5", "C12"] and pd.isna(out[4])
    assert sp.category_labels()["C7"] == "C7 · DEF 29–32"


def test_value_targets_and_zero_value():
    df = pd.DataFrame({"tm_value_T": [10e6, 0.0], "tm_value_T1": [15e6, 1e6], "tm_value_T2": [5e6, np.nan]})
    out = sp.add_value_targets(df)
    assert out.loc[0, "delta_v_T1"] == 5e6 and out.loc[0, "delta_v_pct_T1"] == 50
    assert np.isclose(out.loc[0, "log_return_T2"], np.log(0.5))
    assert out.loc[1, ["delta_v_T1", "delta_v_pct_T1", "log_return_T1"]].isna().all()


def _raw():
    rows = []
    for i in range(20):
        rows.append({"season": "2324", "is_lag_only": False, "fbref_player_id": f"p{i}", "fbref_team_id": "t",
                     "fb_minutes": 1000.0, "fb_position": "MF", "fb_birth_year": 2000, "is_goalkeeper": False,
                     "tm_player_id": float(i), "tm_position": "Central Midfield", "tm_birth_date": "2000-01-01",
                     "tm_value_T": float((i + 1) * 1e6), "tm_value_T1": float((i + 1) * 1e6),
                     "tm_value_T2": np.nan})
    df = pd.DataFrame(rows)
    df.loc[0, "fb_minutes"] = 50.0                      # cai na minutagem
    df.loc[1, ["is_goalkeeper", "tm_position"]] = [True, "Goalkeeper"]
    df.loc[2, "tm_player_id"] = np.nan                  # sem match
    # transferido: duas linhas que somam mais de 90 minutos
    extra = df.loc[[3]].assign(fbref_team_id="t2", fb_minutes=40.0)
    df.loc[3, "fb_minutes"] = 60.0
    lag = df.loc[[4]].assign(season="2223", is_lag_only=True)
    return pd.concat([df, extra, lag], ignore_index=True)


def test_build_analytic_sample_funnel():
    res = sp.build_analytic_sample(_raw())
    sample, funnel, thr = res.sample, res.funnel, res.thresholds
    obs = dict(zip(funnel["etapa"], funnel["observacoes"]))
    assert obs["Registros FBref (jogador–clube–temporada)"] == 21
    assert obs["Jogador–temporada"] == 20
    assert obs["Pareados com o Transfermarkt"] == 19
    assert obs["Exclusão de goleiros"] == 18
    assert obs["Minutagem ≥ 90 min em T"] == 17          # p3 somado (100 min) permanece
    assert "p3" in set(sample["fbref_player_id"]) or thr.loc["2324"] < 4e6
    assert (sample["tm_value_T"] <= thr.loc["2324"]).all()
    assert len(sample) == obs["Corte p90 de V_T"] < 17
    assert set(sample["category"]) == {"C5"}  # 24 anos em 30/06/2024
    assert res.universe["in_sample"].sum() == len(sample) and len(res.universe) == 20
