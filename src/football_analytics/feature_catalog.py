"""Catálogo das variáveis do dataset processado: papel, grupo, fonte, unidade e descrição.

É a fonte única do dicionário de dados do artigo e dos conjuntos de features que o notebook 04 lê.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Var:
    role: str      # id | meta | feature | categorical | lookahead | target | audit
    group: str
    source: str
    unit: str
    description: str


_P90 = "por 90 min"
CATALOG: dict[str, Var] = {
    # identificação e metadados
    "season": Var("meta", "Identificação", "—", "código", "Temporada de referência T (ex.: 2324 = 2023/24)"),
    "season_order": Var("meta", "Identificação", "—", "índice", "Ordem cronológica de T (0 = 2019/20), para a janela expandida"),
    "fbref_player_id": Var("id", "Identificação", "FBref", "—", "Identificador longitudinal do atleta"),
    "tm_player_id": Var("id", "Identificação", "Transfermarkt", "—", "Identificador do atleta no Transfermarkt"),
    "fb_player": Var("meta", "Identificação", "FBref", "—", "Nome do atleta"),
    "fb_team": Var("meta", "Identificação", "FBref", "—", "Clube principal em T (mais minutos)"),
    "fbref_team_id": Var("id", "Identificação", "FBref", "—", "Identificador do clube principal em T"),
    "value_T": Var("meta", "Mercado", "Transfermarkt", "€", "Valor de mercado em T (V_T), para reconverter previsões"),
    "value_T1": Var("meta", "Mercado", "Transfermarkt", "€", "Valor de mercado em T+1 (V_{T+1})"),
    "value_T2": Var("meta", "Mercado", "Transfermarkt", "€", "Valor de mercado em T+2 (V_{T+2})"),
    # categóricas
    "league": Var("categorical", "Contexto", "FBref", "—", "Liga em T"),
    "position_group": Var("categorical", "Demográfico", "Transfermarkt/FBref", "—", "Posição de linha: DEF, MEI, ATA"),
    "age_band": Var("categorical", "Demográfico", "Transfermarkt", "—", "Faixa etária: 16–23, 24–28, 29–32, 33+"),
    "category": Var("categorical", "Demográfico", "Transfermarkt", "—", "Categoria C1–C12 (faixa etária × posição)"),
    "confederation": Var("categorical", "Demográfico", "Transfermarkt", "—", "Confederação da 1ª nacionalidade"),
    # demográficas
    "age": Var("feature", "Demográfico", "Transfermarkt", "anos", "Idade em anos completos em 30/jun de T"),
    "age_sq": Var("feature", "Demográfico", "Transfermarkt", "anos²", "Idade ao quadrado (concavidade do ciclo de vida)"),
    "height_cm": Var("feature", "Demográfico", "Transfermarkt", "cm", "Altura"),
    "is_left_footed": Var("feature", "Demográfico", "Transfermarkt", "0/1", "Canhoto"),
    "is_two_footed": Var("feature", "Demográfico", "Transfermarkt", "0/1", "Ambidestro"),
    "is_uefa": Var("feature", "Demográfico", "Transfermarkt", "0/1", "1ª nacionalidade da UEFA"),
    "is_conmebol": Var("feature", "Demográfico", "Transfermarkt", "0/1", "1ª nacionalidade da CONMEBOL"),
    "is_caf": Var("feature", "Demográfico", "Transfermarkt", "0/1", "1ª nacionalidade da CAF"),
    "is_afc_concacaf": Var("feature", "Demográfico", "Transfermarkt", "0/1", "1ª nacionalidade da AFC, CONCACAF ou OFC"),
    "has_eu_passport": Var("feature", "Demográfico", "Transfermarkt", "0/1", "Possui nacionalidade da UE (britânicos até 2019/20, pelo Brexit)"),
    "n_nationalities": Var("feature", "Demográfico", "Transfermarkt", "contagem", "Número de nacionalidades"),
    "is_domestic": Var("feature", "Demográfico", "Transfermarkt", "0/1", "Possui a nacionalidade do país da liga"),
    # contratuais
    "contract_years_remaining_T": Var("lookahead", "Contratual", "Transfermarkt (reconstruído)", "anos", "Tempo restante de contrato em 30/jun de T (observado ou estimado)"),
    "bosman_T": Var("lookahead", "Contratual", "Transfermarkt (reconstruído)", "0/1", "Último ano de contrato em T (≤ 1 ano restante)"),
    "tenure_years_T": Var("feature", "Contratual", "Transfermarkt", "anos", "Tempo de casa no clube em T"),
    "is_on_loan_T": Var("feature", "Contratual", "Transfermarkt", "0/1", "Atuava emprestado em T"),
    # mercado e clube
    "log_value_T": Var("feature", "Mercado", "Transfermarkt", "ln €", "Logaritmo do valor de mercado em T"),
    "log_club_value_T": Var("feature", "Mercado", "Transfermarkt", "ln €", "Logaritmo da soma de V_T do elenco do clube principal"),
    "value_share_of_club": Var("feature", "Mercado", "Transfermarkt", "proporção", "V_T do atleta sobre o valor do elenco"),
    "club_value_rank_pct": Var("feature", "Mercado", "Transfermarkt", "percentil", "Posição do valor do elenco entre os clubes das Big Five na temporada"),
    "club_value_rank_in_league": Var("feature", "Mercado", "Transfermarkt", "posição", "Posição do valor do elenco dentro da liga (1 = mais valioso)"),
    "team_points_per_game": Var("feature", "Contexto", "FBref", "pontos/jogo", "Pontos por jogo do clube nas partidas do atleta"),
    "market_log_return_lag1_T": Var("feature", "Contexto", "Transfermarkt", "ln", "Regime de mercado: log-retorno médio (aparado a 10%) T−1→T dos atletas das Big Five"),
    "league_log_return_lag1_T": Var("feature", "Contexto", "Transfermarkt", "ln", "Regime da liga: log-retorno médio (aparado a 10%) T−1→T dos atletas da liga"),
    # volume
    "minutes": Var("feature", "Volume", "FBref", "minutos", "Minutos jogados em T (soma entre clubes)"),
    "minutes_share_pct": Var("feature", "Volume", "FBref", "%", "Minutos jogados sobre os minutos possíveis"),
    "games": Var("feature", "Volume", "FBref", "jogos", "Partidas disputadas"),
    "starts_share_pct": Var("feature", "Volume", "FBref", "%", "Partidas como titular sobre partidas disputadas"),
    "complete_games_share_pct": Var("feature", "Volume", "FBref", "%", "Partidas completas sobre partidas disputadas"),
    "n_clubs_big5": Var("feature", "Volume", "FBref", "contagem", "Clubes das Big Five defendidos em T"),
    # ofensivo
    "goals_p90": Var("feature", "Ofensivo", "FBref", _P90, "Gols"),
    "assists_p90": Var("feature", "Ofensivo", "FBref", _P90, "Assistências"),
    "npg_p90": Var("feature", "Ofensivo", "FBref", _P90, "Gols sem pênaltis"),
    "goals_assists_p90": Var("feature", "Ofensivo", "FBref", _P90, "Gols + assistências"),
    "shots_p90": Var("feature", "Ofensivo", "FBref", _P90, "Finalizações"),
    "shots_on_target_p90": Var("feature", "Ofensivo", "FBref", _P90, "Finalizações no alvo"),
    "shot_accuracy_pct": Var("feature", "Ofensivo", "FBref", "%", "Finalizações no alvo sobre finalizações"),
    "shot_conversion_pct": Var("feature", "Ofensivo", "FBref", "%", "Gols sobre finalizações"),
    "xg_p90": Var("feature", "Ofensivo", "Understat", _P90, "Gols esperados (xG)"),
    "npxg_p90": Var("feature", "Ofensivo", "Understat", _P90, "xG sem pênaltis"),
    "xa_p90": Var("feature", "Ofensivo", "Understat", _P90, "Assistências esperadas (xA)"),
    "xg_xa_p90": Var("feature", "Ofensivo", "Understat", _P90, "xG + xA"),
    "xg_per_shot": Var("feature", "Ofensivo", "Understat", "xG/chute", "Qualidade média das finalizações"),
    "finishing_overperf_p90": Var("feature", "Ofensivo", "Understat", _P90, "Gols sem pênalti menos npxG"),
    "big_chances_missed_p90": Var("feature", "Ofensivo", "Sofascore", _P90, "Grandes chances desperdiçadas"),
    "offsides_p90": Var("feature", "Ofensivo", "FBref", _P90, "Impedimentos"),
    # criação e posse
    "key_passes_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Passes para finalização"),
    "big_chances_created_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Grandes chances criadas"),
    "xg_chain_p90": Var("feature", "Criação e posse", "Understat", _P90, "xG das posses com participação do atleta"),
    "xg_buildup_p90": Var("feature", "Criação e posse", "Understat", _P90, "xGChain excluindo finalização e passe final"),
    "passes_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Passes tentados"),
    "accurate_passes_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Passes certos"),
    "pass_accuracy_pct": Var("feature", "Criação e posse", "Sofascore", "%", "Taxa de acerto de passes"),
    "long_balls_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Bolas longas certas"),
    "long_ball_accuracy_pct": Var("feature", "Criação e posse", "Sofascore", "%", "Acerto de bolas longas"),
    "final_third_passes_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Passes certos no terço final"),
    "crosses_p90": Var("feature", "Criação e posse", "FBref", _P90, "Cruzamentos"),
    "accurate_crosses_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Cruzamentos certos"),
    "dribbles_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Dribles certos"),
    "dribble_success_pct": Var("feature", "Criação e posse", "Sofascore", "%", "Taxa de sucesso no drible"),
    "touches_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Toques na bola"),
    "possession_lost_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Perdas de posse"),
    "dispossessed_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Desarmado pelo adversário"),
    "fouled_p90": Var("feature", "Criação e posse", "FBref", _P90, "Faltas sofridas"),
    # defensivo
    "tackles_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Desarmes"),
    "tackles_won_p90": Var("feature", "Defensivo", "FBref", _P90, "Desarmes com recuperação da posse"),
    "interceptions_p90": Var("feature", "Defensivo", "FBref", _P90, "Interceptações"),
    "tackles_interceptions_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Desarmes + interceptações"),
    "clearances_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Cortes"),
    "dribbled_past_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Vezes driblado"),
    "aerial_duels_won_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Duelos aéreos ganhos"),
    "aerial_duels_won_pct": Var("feature", "Defensivo", "Sofascore", "%", "Taxa de duelos aéreos ganhos"),
    "ground_duels_won_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Duelos no chão ganhos"),
    "ground_duels_won_pct": Var("feature", "Defensivo", "Sofascore", "%", "Taxa de duelos no chão ganhos"),
    "duels_won_pct": Var("feature", "Defensivo", "Sofascore", "%", "Taxa de duelos ganhos"),
    "errors_p90": Var("feature", "Defensivo", "Sofascore", _P90, "Erros que levaram a finalização ou gol"),
    "on_goals_against_p90": Var("feature", "Defensivo", "FBref", _P90, "Gols sofridos pela equipe com o atleta em campo (proxy de clean sheets)"),
    "plus_minus_p90": Var("feature", "Defensivo", "FBref", _P90, "Saldo de gols da equipe com o atleta em campo"),
    # disciplina
    "yellow_cards_p90": Var("feature", "Disciplina", "FBref", _P90, "Cartões amarelos"),
    "red_cards_p90": Var("feature", "Disciplina", "FBref", _P90, "Cartões vermelhos"),
    "fouls_p90": Var("feature", "Disciplina", "FBref", _P90, "Faltas cometidas"),
    # avaliação agregada
    "sofascore_rating": Var("feature", "Avaliação agregada", "Sofascore", "nota 0–10", "Nota média do Sofascore, ponderada por minutos"),
    # defasagens
    "log_return_lag1": Var("feature", "Defasagem", "Transfermarkt", "ln", "Log-retorno do valor entre T−1 e T (momentum)"),
    "log_return_lag2": Var("feature", "Defasagem", "Transfermarkt", "ln", "Log-retorno do valor entre T−2 e T"),
    "mkt_log_return_6m": Var("feature", "Dinâmica de mercado", "Transfermarkt", "ln", "Log-retorno do valor nos 6 meses até T"),
    "mkt_last_change": Var("feature", "Dinâmica de mercado", "Transfermarkt", "ln", "Tamanho da última revisão de valor até T"),
    "mkt_days_since_change": Var("feature", "Dinâmica de mercado", "Transfermarkt", "dias", "Dias desde a última revisão de valor"),
    "mkt_n_revisions_12m": Var("feature", "Dinâmica de mercado", "Transfermarkt", "contagem", "Avaliações publicadas nos 12 meses até T"),
    "mkt_n_up_24m": Var("feature", "Dinâmica de mercado", "Transfermarkt", "contagem", "Revisões para cima nos 24 meses até T"),
    "mkt_n_down_24m": Var("feature", "Dinâmica de mercado", "Transfermarkt", "contagem", "Revisões para baixo nos 24 meses até T"),
    "mkt_volatility_24m": Var("feature", "Dinâmica de mercado", "Transfermarkt", "ln", "Desvio-padrão das revisões nos 24 meses até T"),
    "mkt_peak_gap": Var("feature", "Dinâmica de mercado", "Transfermarkt", "ln", "Distância ao pico histórico: ln(V_T / máximo até T)"),
    "fair_log_value_T": Var("feature", "Valor justo", "Calculada", "ln €", "Valor justo em T: ln(V) previsto pelo desempenho (corte transversal de T, validação cruzada)"),
    "value_gap_T": Var("feature", "Valor justo", "Calculada", "ln", "Gap de avaliação: ln(V_T) − valor justo (negativo = subavaliado)"),
    "has_big5_lag1": Var("feature", "Defasagem", "FBref", "0/1", "Atuou nas Big Five em T−1"),
    "lag1_minutes": Var("feature", "Defasagem", "FBref", "minutos", "Minutos jogados em T−1"),
    "lag1_minutes_share_pct": Var("feature", "Defasagem", "FBref", "%", "Participação nos minutos em T−1"),
    "lag1_sofascore_rating": Var("feature", "Defasagem", "Sofascore", "nota", "Nota média em T−1"),
    "lag1_goals_assists_p90": Var("feature", "Defasagem", "FBref", _P90, "Gols + assistências em T−1"),
    "lag1_xg_xa_p90": Var("feature", "Defasagem", "Understat", _P90, "xG + xA em T−1"),
    "lag1_key_passes_p90": Var("feature", "Defasagem", "Sofascore", _P90, "Passes para finalização em T−1"),
    "lag1_dribbles_p90": Var("feature", "Defasagem", "Sofascore", _P90, "Dribles certos em T−1"),
    "lag1_tackles_interceptions_p90": Var("feature", "Defasagem", "Sofascore", _P90, "Desarmes + interceptações em T−1"),
    "lag1_aerial_duels_won_p90": Var("feature", "Defasagem", "Sofascore", _P90, "Duelos aéreos ganhos em T−1"),
    "lag1_pass_accuracy_pct": Var("feature", "Defasagem", "Sofascore", "%", "Acerto de passes em T−1"),
    "delta_minutes_share_pct": Var("feature", "Defasagem", "FBref", "p.p.", "Variação da participação nos minutos (T − T−1)"),
    "delta_sofascore_rating": Var("feature", "Defasagem", "Sofascore", "nota", "Variação da nota média (T − T−1)"),
    # alvos
    "delta_v_T1": Var("target", "Alvo", "Transfermarkt", "€", "ΔV em T+1: V_{T+1} − V_T"),
    "delta_v_pct_T1": Var("target", "Alvo", "Transfermarkt", "%", "ΔV% em T+1: (V_{T+1} − V_T)/V_T × 100"),
    "log_return_T1": Var("target", "Alvo", "Transfermarkt", "ln", "Log-retorno em T+1: ln(V_{T+1}/V_T) — alvo de treino recomendado"),
    "delta_v_T2": Var("target", "Alvo", "Transfermarkt", "€", "ΔV em T+2: V_{T+2} − V_T"),
    "delta_v_pct_T2": Var("target", "Alvo", "Transfermarkt", "%", "ΔV% em T+2: (V_{T+2} − V_T)/V_T × 100"),
    "log_return_T2": Var("target", "Alvo", "Transfermarkt", "ln", "Log-retorno em T+2: ln(V_{T+2}/V_T) — alvo de treino recomendado"),
    "has_target_T1": Var("meta", "Alvo", "Transfermarkt", "0/1", "V_{T+1} observado"),
    "has_target_T2": Var("meta", "Alvo", "Transfermarkt", "0/1", "V_{T+2} observado"),
    # auditoria (não usar como feature)
    "contract_method": Var("audit", "Auditoria", "Transfermarkt", "—", "Como o contrato em T foi obtido (usa informação posterior a T)"),
    "contract_is_estimated": Var("audit", "Auditoria", "Transfermarkt", "0/1", "Contrato em T estimado (usa informação posterior a T)"),
    "contract_expiry_T": Var("audit", "Auditoria", "Transfermarkt", "data", "Término do contrato vigente em T, quando observado"),
    "birth_date_is_estimated": Var("audit", "Auditoria", "FBref", "0/1", "Nascimento aproximado pelo ano do FBref"),
    "tm_value_date_T": Var("audit", "Auditoria", "Transfermarkt", "data", "Data da avaliação usada como V_T"),
}


# Rótulos curtos para gráficos, quando a descrição é longa demais para um eixo.
SHORT_LABELS: dict[str, str] = {
    "xg_chain_p90": "xGChain /90", "xg_buildup_p90": "xGBuildup /90", "sofascore_rating": "Nota Sofascore",
    "on_goals_against_p90": "Gols sofridos em campo /90", "plus_minus_p90": "Saldo de gols em campo /90",
    "finishing_overperf_p90": "Gols − npxG /90", "shot_accuracy_pct": "% finalizações no alvo",
    "shot_conversion_pct": "% conversão de finalizações", "xg_per_shot": "xG por finalização",
    "errors_p90": "Erros graves /90", "tackles_won_p90": "Desarmes ganhos (FBref) /90",
    "xg_p90": "xG /90", "npxg_p90": "npxG /90", "xa_p90": "xA /90",
    "big_chances_missed_p90": "Grandes chances perdidas /90",
    "age": "Idade", "age_sq": "Idade²", "height_cm": "Altura", "has_eu_passport": "Passaporte UE",
    "is_domestic": "Atleta nacional", "tenure_years_T": "Tempo de casa", "is_on_loan_T": "Emprestado em T",
    "log_value_T": "log V_T", "value_gap_T": "Gap de avaliação", "fair_log_value_T": "Valor justo (log)",
    "value_share_of_club": "% do valor do elenco", "log_club_value_T": "log valor do elenco",
    "club_value_rank_pct": "Ranking de valor do elenco (Big Five)", "club_value_rank_in_league": "Ranking de valor do elenco (liga)",
    "team_points_per_game": "Pontos/jogo da equipe", "market_log_return_lag1_T": "Regime de mercado",
    "league_log_return_lag1_T": "Regime da liga", "minutes": "Minutos", "minutes_share_pct": "% dos minutos possíveis",
    "games": "Partidas", "starts_share_pct": "% como titular", "complete_games_share_pct": "% de jogos completos",
    "n_clubs_big5": "Nº de clubes (Big Five)", "mkt_log_return_6m": "Variação do valor em 6 meses",
    "mkt_last_change": "Última revisão de valor", "mkt_days_since_change": "Dias desde a última revisão",
    "mkt_n_revisions_12m": "Nº de revisões (12 meses)", "mkt_n_up_24m": "Revisões para cima (24 meses)",
    "mkt_n_down_24m": "Revisões para baixo (24 meses)", "mkt_volatility_24m": "Volatilidade das revisões (24 meses)",
    "mkt_peak_gap": "Distância ao pico de valor", "log_return_lag1": "Momentum T−1→T", "log_return_lag2": "Momentum T−2→T",
    "delta_minutes_share_pct": "Δ % dos minutos (T−1→T)", "delta_sofascore_rating": "Δ nota Sofascore (T−1→T)",
    "has_big5_lag1": "Jogou nas Big Five em T−1", "contract_years_remaining_T": "Anos de contrato restantes",
    "bosman_T": "Último ano de contrato",
}


def label(name: str) -> str:
    """Rótulo legível de uma variável: curto se houver, senão a descrição (+ "/90" nas taxas)."""
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    var = CATALOG[name]
    return var.description.split(" (")[0] + (" /90" if var.unit == _P90 else "")


def by_role(role: str) -> list[str]:
    return [name for name, v in CATALOG.items() if v.role == role]


def dictionary_frame(columns: list[str] | None = None) -> pd.DataFrame:
    rows = [{"variavel": k, "papel": v.role, "grupo": v.group, "fonte": v.source, "unidade": v.unit,
             "descricao": v.description} for k, v in CATALOG.items() if columns is None or k in columns]
    return pd.DataFrame(rows)
