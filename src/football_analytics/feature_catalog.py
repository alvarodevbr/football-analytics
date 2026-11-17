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
    "season_order": Var("meta", "Identificação", "—", "índice", "Ordem cronológica de T (0 = 2018/19), para a janela expandida"),
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
    "role": Var("categorical", "Demográfico", "Transfermarkt", "—", "Papel tático na temporada: zagueiro, lateral, volante, meia central, meia ofensivo ou ponta, centroavante"),
    "age_band": Var("categorical", "Demográfico", "Transfermarkt", "—", "Faixa etária: 16–23, 24–28, 29–32, 33+"),
    "category": Var("categorical", "Demográfico", "Transfermarkt", "—", "Categoria C1–C12 (faixa etária × posição)"),
    "age_band_3x3": Var("categorical", "Demográfico", "Transfermarkt", "—", "Faixa etária da matriz 3 × 3 de Khalife et al. (2025): 16–23, 24–29, 30+"),
    "category_3x3": Var("categorical", "Demográfico", "Transfermarkt", "—", "Categoria K1–K9 da matriz 3 × 3 (faixa etária × posição)"),
    "confederation": Var("categorical", "Demográfico", "Transfermarkt", "—", "Confederação da 1ª nacionalidade"),
    # demográficas
    "age": Var("feature", "Demográfico", "Transfermarkt", "anos", "Idade em anos, com fração decimal, em 30/jun de T"),
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
    "days_since_valuation_T": Var("feature", "Dinâmica de mercado", "Transfermarkt", "dias", "Dias entre a avaliação usada como V_T e 30/jun de T"),
    "log_club_value_T": Var("feature", "Mercado", "Transfermarkt", "ln €", "Logaritmo da soma de V_T do elenco do clube principal"),
    "value_share_of_club": Var("feature", "Mercado", "Transfermarkt", "proporção", "V_T do atleta sobre o valor do elenco"),
    "club_value_rank_pct": Var("feature", "Mercado", "Transfermarkt", "percentil", "Posição do valor do elenco entre os clubes das Big Five na temporada"),
    "club_value_rank_in_league": Var("feature", "Mercado", "Transfermarkt", "posição", "Posição do valor do elenco dentro da liga (1 = mais valioso)"),
    "team_points_per_game": Var("feature", "Contexto", "FBref", "pontos/jogo", "Pontos por jogo do clube nas partidas do atleta"),
    "market_log_return_lag1_T": Var("feature", "Contexto", "Transfermarkt", "ln", "Regime de mercado: log-retorno médio (aparado a 10%) T−1→T dos atletas das Big Five"),
    "league_log_return_lag1_T": Var("feature", "Contexto", "Transfermarkt", "ln", "Regime da liga: log-retorno médio (aparado a 10%) T−1→T dos atletas da liga"),
    "league_rank_pct_T": Var("feature", "Contexto", "Transfermarkt", "0–1", "Posição final do clube na liga em T, normalizada (0 = campeão, 1 = último)"),
    "league_gd_per_match_T": Var("feature", "Contexto", "Transfermarkt", "gols/jogo", "Saldo de gols por jogo do clube na liga em T"),
    "promoted_T": Var("feature", "Contexto", "Transfermarkt", "0/1", "Clube recém-promovido: não estava nas Big Five em T−1"),
    "relegated_T": Var("feature", "Contexto", "Transfermarkt", "0/1", "Clube rebaixado ao fim de T (decidido até a âncora; não está nas Big Five em T+1)"),
    "europe_T": Var("feature", "Contexto", "Transfermarkt", "0–3", "Maior competição UEFA disputada em T: 0 nenhuma, 1 Conferência, 2 Liga Europa, 3 Liga dos Campeões"),
    "club_elo_T": Var("feature", "Contexto", "ClubElo", "pontos Elo", "Elo do clube em 30/jun de T"),
    "club_elo_change_T": Var("feature", "Contexto", "ClubElo", "pontos Elo", "Variação do Elo do clube entre 01/ago e 30/jun de T"),
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
    "tail_xg_xa_p90": Var("feature", "Ofensivo", "Understat", "probabilidade", "P(taxa de xG + xA acima do 3º quartil do papel tático | dados): surpresa de cauda"),
    "xg_per_shot": Var("feature", "Ofensivo", "Understat", "xG/chute", "Qualidade média das finalizações"),
    "finishing_overperf_p90": Var("feature", "Ofensivo", "Understat", _P90, "Gols sem pênalti menos npxG"),
    "big_chances_missed_p90": Var("feature", "Ofensivo", "Sofascore", _P90, "Grandes chances desperdiçadas"),
    "offsides_p90": Var("feature", "Ofensivo", "FBref", _P90, "Impedimentos"),
    # criação e posse
    "key_passes_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Passes para finalização"),
    "big_chances_created_p90": Var("feature", "Criação e posse", "Sofascore", _P90, "Grandes chances criadas"),
    "xg_chain_p90": Var("feature", "Criação e posse", "Understat", _P90, "xG das posses com participação do atleta"),
    "xg_buildup_p90": Var("feature", "Criação e posse", "Understat", _P90, "xGChain excluindo finalização e passe final"),
    "xg_buildup_share": Var("feature", "Criação e posse", "Understat", "proporção", "xGBuildup / xGChain: participação na construção, sem chute nem passe final"),
    "tail_key_passes_p90": Var("feature", "Criação e posse", "Sofascore", "probabilidade", "P(taxa de passes para finalização acima do 3º quartil do papel tático | dados)"),
    "tail_dribbles_p90": Var("feature", "Criação e posse", "Sofascore", "probabilidade", "P(taxa de dribles certos acima do 3º quartil do papel tático | dados)"),
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
    "tail_tackles_interceptions_p90": Var("feature", "Defensivo", "Sofascore", "probabilidade", "P(taxa de desarmes + interceptações acima do 3º quartil do papel tático | dados)"),
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
    "tail_sofascore_rating": Var("feature", "Avaliação agregada", "Sofascore", "probabilidade", "P(nota verdadeira acima do 3º quartil do papel tático | dados)"),
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
    "mkt_years_since_peak": Var("feature", "Dinâmica de mercado", "Transfermarkt", "anos", "Anos desde o pico histórico de valor até T"),
    "fair_log_value_T": Var("audit", "Valor justo", "Calculada", "ln €", "Valor justo em T: ln(V) previsto pelo desempenho e pelo contexto (corte transversal de T, validação cruzada agrupada por clube); fora do vetor por ser ln(V_T) − gap"),
    "value_gap_T": Var("feature", "Valor justo", "Calculada", "ln", "Gap de avaliação: ln(V_T) − valor justo (negativo = subavaliado)"),
    "fair_log_value_perf_T": Var("audit", "Valor justo", "Calculada", "ln €", "Valor justo só pelo desempenho, idade e posição (sem liga nem contexto do clube)"),
    "value_gap_perf_T": Var("feature", "Valor justo", "Calculada", "ln", "Gap de avaliação só pelo desempenho: ln(V_T) − valor justo sem liga nem clube"),
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
    "lag2_minutes_share_pct": Var("feature", "Defasagem", "FBref", "%", "Participação nos minutos em T−2"),
    "minutes_share_accel": Var("feature", "Defasagem", "FBref", "p.p.", "Aceleração da participação nos minutos: (T − T−1) − (T−1 − T−2)"),
    "breakout_starter_T": Var("feature", "Defasagem", "FBref", "0/1", "Virou titular: menos de 900 minutos em T−1 e ao menos 1.800 em T"),
    "league_step_up_T": Var("feature", "Defasagem", "FBref", "0/1", "Chegou às Big Five em T e jogou acima da mediana de participação"),
    # forma dentro da temporada (Understat, escalações por partida)
    "h2_minutes_share_T": Var("feature", "Forma intra-temporada", "Understat", "%", "Participação nos minutos do último clube no 2º turno (de 1º/jan ao fim da temporada)"),
    "form_minutes_trend_T": Var("feature", "Forma intra-temporada", "Understat", "p.p.", "Participação no 2º turno menos a do 1º turno"),
    "h2_xg_chain_p90_T": Var("feature", "Forma intra-temporada", "Understat", _P90, "xGChain no 2º turno, encolhido"),
    "post_valuation_minutes_T": Var("feature", "Forma intra-temporada", "Understat", "minutos", "Minutos jogados depois da data da avaliação usada como V_T"),
    "post_valuation_xg_xa_p90_T": Var("feature", "Forma intra-temporada", "Understat", _P90, "xG + xA depois da data de V_T (ainda não incorporados ao valor), encolhido"),
    "debut_second_half_T": Var("feature", "Forma intra-temporada", "Understat", "0/1", "Estreou na temporada só no 2º turno (chegada em janeiro ou entrada tardia)"),
    "club_moved_mid_season_T": Var("feature", "Forma intra-temporada", "Understat", "0/1", "O clube em 30/jun difere do clube com mais minutos na temporada"),
    # disponibilidade física (Transfermarkt, histórico datado de lesões)
    "injury_days_T": Var("feature", "Disponibilidade física", "Transfermarkt", "dias", "Dias lesionado nos 12 meses até 30/jun de T (truncado na âncora)"),
    "injury_days_Tm1": Var("feature", "Disponibilidade física", "Transfermarkt", "dias", "Dias lesionado nos 12 meses anteriores (T−1)"),
    "n_injuries_24m": Var("feature", "Disponibilidade física", "Transfermarkt", "contagem", "Lesões iniciadas nos 24 meses até T"),
    "injured_at_anchor": Var("feature", "Disponibilidade física", "Transfermarkt", "0/1", "Lesionado em 30/jun de T"),
    "long_injury_at_anchor": Var("feature", "Disponibilidade física", "Transfermarkt", "0/1", "Lesão grave ou em curso há 60 dias ou mais em 30/jun de T"),
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
    "value_T1_last_obs": Var("audit", "Auditoria", "Transfermarkt", "€", "Último valor publicado até 30/jun de T+1, sem limite de defasagem (limite otimista do backtest)"),
    "value_T2_last_obs": Var("audit", "Auditoria", "Transfermarkt", "€", "Último valor publicado até 30/jun de T+2, sem limite de defasagem (limite otimista do backtest)"),
    "target_missing_reason_T1": Var("audit", "Auditoria", "Transfermarkt", "—", "Motivo da ausência de V_{T+1}: observed, not_due, career_end ou stale"),
    "target_missing_reason_T2": Var("audit", "Auditoria", "Transfermarkt", "—", "Motivo da ausência de V_{T+2}: observed, not_due, career_end ou stale"),
    "tm_value_date_T1": Var("audit", "Auditoria", "Transfermarkt", "data", "Data da avaliação usada como V_{T+1}"),
    "tm_value_date_T2": Var("audit", "Auditoria", "Transfermarkt", "data", "Data da avaliação usada como V_{T+2}"),
    "log_return_ann_T1": Var("audit", "Alvo", "Transfermarkt", "ln/ano", "Log-retorno anualizado pelos dias exatos entre as avaliações de T e T+1 (robustez)"),
    "log_return_ann_T2": Var("audit", "Alvo", "Transfermarkt", "ln/ano", "Log-retorno de T+2 anualizado pelos dias exatos entre as avaliações (robustez; ×2 para a escala de dois anos)"),
}

# Features cuja fonte pode estar incompleta: saem do vetor se a cobertura for baixa em alguma
# temporada, para não criar um padrão de ausência próprio de uma temporada. Em out/2026, a API do
# ClubElo estava fora do ar e o Transfermarkt bloqueou a coleta do histórico de transferências dos
# atletas novos de 2024/25 (fonte do tempo de casa e do empréstimo em T).
OPTIONAL_FEATURES: tuple[str, ...] = ("club_elo_T", "club_elo_change_T", "tenure_years_T", "is_on_loan_T",
                                      "injury_days_T", "injury_days_Tm1", "n_injuries_24m", "injured_at_anchor",
                                      "long_injury_at_anchor", "h2_minutes_share_T", "form_minutes_trend_T",
                                      "h2_xg_chain_p90_T", "post_valuation_minutes_T", "post_valuation_xg_xa_p90_T",
                                      "debut_second_half_T", "club_moved_mid_season_T")


def _add_raw_twins() -> None:
    """Versões cruas (sem encolhimento empírico-bayesiano) das taxas: papel `audit`, usadas só na
    comparação crua × encolhida do notebook 04."""
    from .features import SHRUNK_FEATURES

    for name in SHRUNK_FEATURES:
        v = CATALOG[name]
        CATALOG[f"{name}_raw"] = Var("audit", v.group, v.source, v.unit,
                                     f"{v.description} — versão crua, sem encolhimento")


_add_raw_twins()


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
    "log_value_T": "Valor de mercado em T (ln)", "value_gap_T": "Gap de avaliação",
    "fair_log_value_T": "Valor justo (ln)", "value_gap_perf_T": "Gap de avaliação (só desempenho)",
    "fair_log_value_perf_T": "Valor justo só desempenho (ln)",
    "value_share_of_club": "% do valor do elenco", "log_club_value_T": "Valor do elenco (ln)",
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
    "days_since_valuation_T": "Dias desde a avaliação de V_T", "mkt_years_since_peak": "Anos desde o pico de valor",
    "xg_buildup_share": "xGBuildup / xGChain", "lag2_minutes_share_pct": "% dos minutos em T−2",
    "minutes_share_accel": "Aceleração da % dos minutos", "breakout_starter_T": "Virou titular em T",
    "league_step_up_T": "Chegada às Big Five com minutos", "injury_days_T": "Dias lesionado (T)",
    "injury_days_Tm1": "Dias lesionado (T−1)", "n_injuries_24m": "Lesões em 24 meses",
    "injured_at_anchor": "Lesionado em 30/jun", "long_injury_at_anchor": "Lesão longa em 30/jun",
    "tail_xg_xa_p90": "P(xG + xA no 3º quartil)", "tail_key_passes_p90": "P(passes decisivos no 3º quartil)",
    "tail_dribbles_p90": "P(dribles no 3º quartil)", "tail_tackles_interceptions_p90": "P(desarmes + interc. no 3º quartil)",
    "tail_sofascore_rating": "P(nota no 3º quartil)", "h2_minutes_share_T": "% dos minutos no 2º turno",
    "form_minutes_trend_T": "Tendência de minutos (2º − 1º turno)", "h2_xg_chain_p90_T": "xGChain /90 no 2º turno",
    "post_valuation_minutes_T": "Minutos após a avaliação de V_T", "post_valuation_xg_xa_p90_T": "xG + xA /90 após a avaliação",
    "debut_second_half_T": "Estreia no 2º turno", "club_moved_mid_season_T": "Trocou de clube na temporada",
}


def label(name: str) -> str:
    """Rótulo legível de uma variável: curto se houver, senão a descrição (+ "/90" nas taxas)."""
    if name.endswith("_raw") and name[:-4] in CATALOG:
        return f"{label(name[:-4])} (cru)"
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    var = CATALOG[name]
    return var.description.split(" (")[0] + (" /90" if var.unit == _P90 else "")


def by_role(role: str) -> list[str]:
    return [name for name, v in CATALOG.items() if v.role == role]


def dictionary_frame(columns: list[str] | None = None) -> pd.DataFrame:
    from .features import SHRUNK_FEATURES

    rows = [{"variavel": k, "papel": v.role, "grupo": v.group, "fonte": v.source, "unidade": v.unit,
             "encolhida": k in SHRUNK_FEATURES, "descricao": v.description}
            for k, v in CATALOG.items() if columns is None or k in columns]
    return pd.DataFrame(rows)
