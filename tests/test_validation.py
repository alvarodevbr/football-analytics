from football_analytics.config import default_config
from football_analytics.validation import expanding_window_folds


def test_folds_respect_target_availability_and_holdouts():
    cfg = default_config()
    f1 = expanding_window_folds(cfg.target_seasons(1), 1)
    f2 = expanding_window_folds(cfg.target_seasons(2), 2)
    assert len(f1) == 6 and f1[0].train == ("1819",) and f1[0].test == "1920"
    assert f1[-1].test == cfg.holdout_seasons[1] == "2425"
    assert len(f2) == 4 and f2[-1].test == cfg.holdout_seasons[2] == "2324"
    # T+2: a temporada imediatamente anterior ao teste é purgada (seu alvo seria medido depois de T)
    for fold in f2:
        assert len(fold.purged) == 1 and fold.purged[0] not in fold.train
        assert max(fold.train) < fold.purged[0] < fold.test
    # nenhuma dobra de T+2 testa em 2024/25, cujo alvo (30/06/2027) ainda não existe
    assert all(f.test != "2425" for f in f2)
