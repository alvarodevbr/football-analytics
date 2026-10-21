"""Partições da validação por janela expandida, com purga para o horizonte T+2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    train: tuple[str, ...]
    test: str
    purged: tuple[str, ...]


def expanding_window_folds(seasons: Sequence[str], horizon: int, min_train: int = 1) -> list[Fold]:
    """Dobras cronológicas: treino em [T_1..T_k], teste em T_{k+horizon}.

    Para k = 2, a temporada T_{k+1} é purgada: o alvo de uma observação de treino em T_{k+1} seria
    medido em T_{k+3}, depois da data em que as features do teste estão disponíveis.
    """
    ordered = sorted(seasons)
    folds = []
    for k in range(min_train, len(ordered) - horizon + 1):
        train = tuple(ordered[:k])
        test = ordered[k + horizon - 1]
        purged = tuple(ordered[k:k + horizon - 1])
        folds.append(Fold(train, test, purged))
    return folds
