# Tabelas manuais da resolução de entidades

Versionadas e editáveis. São lidas pelo notebook 01 (`football_analytics.entity_resolution`).

- `team_aliases.csv`: `source,team,target_team`. Força o mapeamento de um clube do FBref (`team`)
  para o nome na fonte `source` (`transfermarkt`, `understat` ou `sofascore`), dentro de cada
  liga-temporada. O alias só é usado se o alvo existir naquele bloco. Os demais clubes são
  atribuídos 1-para-1 pelo algoritmo húngaro. Revise os avisos "baixa confiança" no log.
- `manual_player_overrides.csv`: `fbref_player_id,tm_player_id,note`. Prevalece sobre qualquer
  match automático, em todas as temporadas do jogador. Use para os casos listados no relatório
  de não casados.
