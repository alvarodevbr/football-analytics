from dataclasses import replace

from football_analytics.config import Paths, Season, default_config
from football_analytics.io import write_json
from football_analytics.sofascore import SofascoreCollector


def test_collect_reads_cache_even_with_zero_fetch_limit(tmp_path):
    """Com `max_fetches_per_run=0` (reexecução offline), o que está em cache é lido; só a rede é bloqueada."""
    cfg = default_config()
    cfg = replace(cfg, paths=Paths(cfg.paths.root, data_root=tmp_path), max_fetches_per_run=0,
                  leagues=cfg.leagues[:1], feature_seasons=("2324",), lag_only_seasons=(), sofascore_fields=("rating",))
    league, season = cfg.leagues[0], Season("2324")
    cache = cfg.paths.sofascore
    tid = league.sofascore_tournament_id
    write_json(cache / "seasons" / f"{tid}.json", {"seasons": [{"year": season.sofascore_year, "id": 99}]})
    write_json(cache / "teams" / f"{tid}_99.json", {"teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]})
    write_json(cache / "stats" / f"{tid}_99_1.json",
               {"results": [{"player": {"id": 10, "name": "x"}, "rating": 7.1}], "pages": 1})
    df = SofascoreCollector(cfg).collect(cfg.leagues, [season])
    # o clube 1 está em cache e é lido; o clube 2 exigiria rede e fica de fora sem abrir o navegador
    assert df["sofascore_team_id"].tolist() == [1]
    assert df["ss_rating"].tolist() == [7.1]
