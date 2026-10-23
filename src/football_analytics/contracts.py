"""Contrato vigente na data-âncora de T, reconstruído a partir do perfil atual e do histórico de
transferências do Transfermarkt.

O Transfermarkt só publica o contrato **atual**. O vínculo vigente em T é observado em dois casos:

* ``current_contract``: o atleta continua no clube de T, e o contrato atual foi assinado (chegada
  ou última renovação) até a âncora. É o mesmo contrato que vigorava em T.
* ``free_exit``: o atleta deixou o clube de T depois da âncora em transferência livre (ou ficou
  sem clube/se aposentou). O contrato de T terminou nessa saída.

Nos demais casos (venda após T, renovação posterior a T, empréstimo em T, perfil sem contrato) o
tempo restante é **estimado** pela mediana observada no mesmo estrato de tempo de casa × faixa
etária, informação disponível em T. O método fica numa coluna de auditoria que **não deve** entrar
como feature: o motivo de o contrato não ser observado (uma venda ou renovação futura) é
informação posterior a T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YEAR = pd.Timedelta(days=365.25)
# FIFA RSTP art. 18.2: contrato profissional de no máximo cinco anos (folga de meio ano para datas).
MAX_CONTRACT = 5.5 * YEAR
WITHOUT_CLUB_IDS = {515, 123}  # "Without Club", "Retired"
NO_CLUB_NAMES = {"Without Club", "Retired", "Career break"}

OBSERVED_METHODS = ("current_contract", "free_exit")
ESTIMATED_METHODS = ("extension_after_T", "transfer_exit", "on_loan", "unrecorded_extension", "no_contract_data",
                     "not_collected")
METHOD_LABELS: dict[str, str] = {
    "current_contract": "Observado: contrato atual já vigente em T",
    "free_exit": "Observado: saída livre após T",
    "extension_after_T": "Estimado: renovação posterior a T",
    "transfer_exit": "Estimado: venda após T",
    "on_loan": "Estimado: emprestado em T",
    "unrecorded_extension": "Estimado: renovação não registrada",
    "no_contract_data": "Estimado: sem dado contratual",
    "not_collected": "Sem coleta: histórico de transferências não coletado (bloqueio do Transfermarkt)",
}


@dataclass(frozen=True)
class ContractAtT:
    club_id: float
    tenure_start: pd.Timestamp | None
    on_loan: bool | float              # NaN quando o histórico de transferências não foi coletado
    expiry: pd.Timestamp | None
    method: str


def _normalize_exit_date(d: pd.Timestamp) -> pd.Timestamp:
    """Saídas livres de verão registradas em julho/agosto correspondem ao fim do contrato em 30/jun."""
    if d.month in (7, 8, 9):
        return pd.Timestamp(d.year, 6, 30)
    return d


def _is_free(move: pd.Series) -> bool:
    return (move["fee_text"] == "free transfer" or move["to_club_id"] in WITHOUT_CLUB_IDS
            or move["to_club"] in NO_CLUB_NAMES)


def contract_for(anchor: pd.Timestamp, moves: pd.DataFrame, profile: pd.Series | None) -> ContractAtT:
    """Contrato vigente em `anchor` para um atleta (`moves` ordenado por data, sem transferências futuras)."""
    past = moves[moves["transfer_date"] <= anchor]
    if past.empty:
        return ContractAtT(np.nan, None, False, None, "no_contract_data")
    last = past.iloc[-1]
    club = last["to_club_id"]
    on_loan = bool(last["is_loan"])
    # início do vínculo permanente com o clube: última chegada que não é empréstimo nem retorno de empréstimo
    arrivals = past[(past["to_club_id"] == club) & ~past["is_loan"] & ~past["is_end_of_loan"]]
    tenure_start = arrivals["transfer_date"].iloc[-1] if not arrivals.empty else last["transfer_date"]
    if club in WITHOUT_CLUB_IDS:
        return ContractAtT(club, tenure_start, False, None, "no_contract_data")
    if on_loan:
        return ContractAtT(club, last["transfer_date"], True, None, "on_loan")

    future = moves[moves["transfer_date"] > anchor]
    exits = future[(future["from_club_id"] == club) & ~future["is_loan"]]
    if not exits.empty:
        exit_move = exits.iloc[0]
        d = exit_move["transfer_date"]
        if _is_free(exit_move) and d - anchor <= MAX_CONTRACT:
            return ContractAtT(club, tenure_start, False, _normalize_exit_date(d), "free_exit")
        return ContractAtT(club, tenure_start, False, None, "transfer_exit")

    if profile is None:
        return ContractAtT(club, tenure_start, False, None, "no_contract_data")
    loaned_out = isinstance(profile.get("on_loan_from"), str) and bool(profile.get("on_loan_from"))
    expiry = profile.get("parent_contract_expires") if loaned_out else profile.get("contract_expires")
    expiry = pd.to_datetime(expiry, errors="coerce")
    extension = pd.to_datetime(profile.get("last_contract_extension"), errors="coerce")
    if pd.isna(expiry):
        return ContractAtT(club, tenure_start, False, None, "no_contract_data")
    if pd.notna(extension) and extension > anchor:
        return ContractAtT(club, tenure_start, False, None, "extension_after_T")
    signed = extension if pd.notna(extension) else tenure_start
    if expiry <= anchor or expiry - signed > MAX_CONTRACT:
        return ContractAtT(club, tenure_start, False, None, "unrecorded_extension")
    return ContractAtT(club, tenure_start, False, expiry, "current_contract")


def tenure_bucket(years: pd.Series) -> pd.Series:
    return pd.cut(years, [-np.inf, 1, 2, 3, 4, np.inf], labels=["<1", "1–2", "2–3", "3–4", "4+"], right=False)


def contracts_at_anchor(
    rows: pd.DataFrame, transfers: pd.DataFrame, profiles: pd.DataFrame, fetched: set[int] | None = None
) -> pd.DataFrame:
    """Contrato em T para cada linha (``tm_player_id``, ``anchor_date_T``, ``age_band``).

    Devolve, no mesmo índice: clube e tempo de casa em T, empréstimo, data de término, anos
    restantes, método (auditoria), flag de estimativa e status Bosman (último ano de contrato).
    `fetched` é o conjunto de atletas cujo histórico de transferências foi coletado: sem ele, um
    atleta não coletado pareceria ter "nenhuma transferência" (tempo de casa ausente e empréstimo
    igual a zero); com ele, tempo de casa e empréstimo ficam ausentes (método ``not_collected``).
    """
    tr = transfers.copy()
    tr["transfer_date"] = pd.to_datetime(tr["transfer_date"], errors="coerce")
    tr = tr[~tr["upcoming"].fillna(False).astype(bool)].dropna(subset=["transfer_date"])
    tr = tr.sort_values(["tm_player_id", "transfer_date"])
    by_player = {pid: g for pid, g in tr.groupby("tm_player_id")}
    prof = profiles.drop_duplicates("tm_player_id").set_index("tm_player_id")

    records = []
    for idx, pid, anchor in zip(rows.index, rows["tm_player_id"], rows["anchor_date_T"]):
        if pd.isna(pid):
            records.append((idx, ContractAtT(np.nan, None, False, None, "no_contract_data")))
            continue
        pid = int(pid)
        if fetched is not None and pid not in fetched:
            records.append((idx, ContractAtT(np.nan, None, np.nan, None, "not_collected")))
            continue
        moves = by_player.get(pid, tr.iloc[0:0])
        profile = prof.loc[pid] if pid in prof.index else None
        records.append((idx, contract_for(anchor, moves, profile)))

    out = pd.DataFrame(
        [{"contract_club_id_T": c.club_id, "tenure_start_T": c.tenure_start, "is_on_loan_T": float(c.on_loan),
          "contract_expiry_T": c.expiry, "contract_method": c.method} for _, c in records],
        index=[i for i, _ in records],
    )
    anchor = rows["anchor_date_T"]
    out["tenure_start_T"] = pd.to_datetime(out["tenure_start_T"])
    out["contract_expiry_T"] = pd.to_datetime(out["contract_expiry_T"])
    out["tenure_years_T"] = (anchor - out["tenure_start_T"]) / YEAR
    out["contract_years_observed_T"] = (out["contract_expiry_T"] - anchor) / YEAR
    out["contract_is_estimated"] = ~out["contract_method"].isin(OBSERVED_METHODS)

    # Estimativa: mediana dos anos restantes observados no mesmo estrato tempo de casa × faixa etária.
    strata = pd.DataFrame({"tenure": tenure_bucket(out["tenure_years_T"]), "band": rows["age_band"].astype(str)},
                          index=out.index)
    observed = out["contract_years_observed_T"].where(~out["contract_is_estimated"])
    medians = observed.groupby([strata["tenure"], strata["band"]], observed=True).median()
    by_band = observed.groupby(strata["band"]).median()
    keys = list(zip(strata["tenure"], strata["band"]))
    est = pd.Series([medians.get(k, np.nan) for k in keys], index=out.index)
    est = est.fillna(strata["band"].map(by_band)).fillna(observed.median())
    out["contract_years_remaining_T"] = observed.fillna(est)
    out["bosman_T"] = (out["contract_years_remaining_T"] <= 1.0 + 1e-6).astype(float)
    return out


def estimator_backtest(out: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Erro da regra de estimativa nos casos observados (validação cruzada por temporada: a mediana
    de cada estrato é calculada sem a temporada avaliada)."""
    obs = out[~out["contract_is_estimated"]].copy()
    obs["tenure"] = tenure_bucket(obs["tenure_years_T"])
    obs["band"] = rows.loc[obs.index, "age_band"].astype(str)
    obs["season"] = rows.loc[obs.index, "season"]
    preds = []
    for season, test in obs.groupby("season"):
        train = obs[obs["season"] != season]
        med = train.groupby(["tenure", "band"], observed=True)["contract_years_observed_T"].median()
        p = pd.Series([med.get(k, np.nan) for k in zip(test["tenure"], test["band"])], index=test.index)
        preds.append(p.fillna(train["contract_years_observed_T"].median()))
    obs["pred"] = pd.concat(preds)
    obs["naive"] = obs["contract_years_observed_T"].median()
    err = obs["pred"] - obs["contract_years_observed_T"]
    naive_err = obs["naive"] - obs["contract_years_observed_T"]
    bosman_true = obs["contract_years_observed_T"] <= 1 + 1e-6
    bosman_pred = obs["pred"] <= 1 + 1e-6
    return pd.Series({
        "n_observados": len(obs),
        "MAE_estrato_anos": err.abs().mean(),
        "MAE_mediana_global_anos": naive_err.abs().mean(),
        "acuracia_bosman": (bosman_true == bosman_pred).mean(),
        "taxa_bosman_observada": bosman_true.mean(),
    })
