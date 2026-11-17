# 01 — Extração e Fusão das Fontes de Dados

**Notebook:** `notebooks/01_data_extraction.ipynb` ·
**Código:** `src/football_analytics/{fbref,understat,sofascore,transfermarkt,http,browser,entity_resolution,valuation,club_context,pipeline}.py`
**Saída:** `data/raw/football_raw_dataset.parquet` (+ `.csv`, `_schema.json`) e tabelas intermediárias em `data/interim/`

[← Índice](README.md)

## 1. Objetivo e escopo

O notebook constrói o dataset bruto consolidado. Cada linha é uma observação **jogador–clube–temporada**
das Big Five, com o desempenho em $T$, atributos demográficos e contratuais, o contexto do clube em $T$
e os valores de mercado em $T$, $T+1$ e $T+2$.

- **Temporadas de referência (versão 3):** 2018/19 a 2024/25. Em 2024/25 só existe o alvo de $T+1$
  (valor em 30/06/2026); o de $T+2$ (30/06/2027) ainda não foi publicado.
- **Temporada adicional:** 2017/18, coletada apenas para as defasagens de desempenho de 2018/19
  (`is_lag_only`).
- **Sem filtros nesta etapa.** O corte $p_{90}$, a minutagem mínima e a exclusão de goleiros são
  validados no notebook 02 e aplicados no 03.
- **Coleta do zero.** Nenhum cache de projetos anteriores foi reutilizado. Todos os caches ficam
  em `data/raw/`.

## 2. Arquitetura de fontes

Em **20/01/2026**, a Sports Reference retirou do FBref todas as métricas avançadas licenciadas pela
Opta, inclusive das temporadas históricas. As páginas *passing*, *possession*, *gca* e *defense*
continuam publicadas, mas com as colunas vazias, como documenta o diagnóstico de preenchimento do
notebook. Para preservar o vetor tático-técnico da especificação, a arquitetura usa quatro fontes:

| Fonte | Acesso | Variáveis | Papel |
|---|---|---|---|
| **FBref** | soccerdata 1.9.1 + Chrome (Cloudflare exige navegador não *headless*) | minutos, jogos, gols, assistências, chutes, cartões, faltas, cruzamentos, interceptações, desarmes ganhos, gols pró/contra em campo, ano de nascimento | Espinha dorsal da amostra e id longitudinal |
| **Understat** | soccerdata (HTTP) | xG, npxG, xA, xGChain, xGBuildup, passes-chave; resultados das partidas | Família de gols esperados e classificação das ligas |
| **Sofascore** | API JSON por clube, consultada dentro do Chrome | passes, bolas longas, terço final, passes-chave, grandes chances, dribles, duelos aéreos e no chão, desarmes, cortes, perdas de posse, nota; clubes das competições UEFA | Vetor tático que o FBref deixou de publicar e participação europeia |
| **Transfermarkt** | `requests` (HTML + `/ceapi`) | valor de mercado datado, nascimento, altura, pé, nacionalidades, posição granular, contrato atual, transferências | Base financeira, alvos e controles |

### Volume coletado

| Tabela | Linhas | Observação |
|---|---|---|
| FBref (jogador–clube–temporada, 8 temporadas) | 22.419 | 4 páginas com dados × 8 temporadas, em páginas combinadas das Big Five |
| Understat | 21.837 | Uma linha por jogador e clube; transferidos aparecem agregados |
| Sofascore | 21.855 | Consulta por clube e temporada |
| Elencos do Transfermarkt | 30.024 | 780 páginas de elenco (clube × temporada) |
| Histórico de valores de mercado | 165.926 avaliações | 6.596 jogadores, endpoint `/ceapi/marketValueDevelopment` |
| Perfis (contrato atual) | 6.554 | Página `/profil/spieler/{id}` |
| Histórico de transferências | 67.606 movimentos | 5.955 jogadores, `/ceapi/transferHistory/list/{id}` (exige os cabeçalhos XHR) |
| Contexto do clube | 682 clubes-temporada | Seção 5 |

Os dados por jogador do Transfermarkt são buscados só para os atletas pareados nas **temporadas de
referência**: a temporada 2017/18 serve apenas às defasagens de desempenho, e o histórico de valores de
cada atleta de referência já cobre o passado dele. Isso poupou cerca de 2 mil requisições.

### Bloqueios durante a coleta e como foram tratados

| Fonte | O que aconteceu (out/2026) | Tratamento | Consequência |
|---|---|---|---|
| Transfermarkt | Captcha do AWS WAF (HTTP 405 com `x-amzn-waf-action`) depois de ~1,4 mil requisições, três dias após um bloqueio anterior (~18 mil requisições, v2) | O cliente HTTP interrompe a rede sem insistir e continua lendo o cache; `tm_fetch_status.parquet` registra quem foi coletado | Faltam 54 perfis e 652 históricos de transferência, quase todos de atletas novos em 2024/25. Tempo de casa e empréstimo saem do vetor no notebook 03 (regra de cobertura) |
| FBref | O desafio do Cloudflare passou a exigir clique; o solucionador do soccerdata depende do `python3-tk`, ausente | O desafio é contornado reabrindo a página com o driver desconectado (`fbref._reconnect_through_challenge`) | Todas as páginas de jogadores de 2017/18 a 2024/25 foram baixadas; classificação e competições UEFA vieram de outras fontes |
| ClubElo | A API respondeu HTTP 502; o site reformulado só publica ~4 anos de histórico | O Elo é opcional: a coleta segue sem ele | Elo fora do vetor (seção 5) |

## 3. Resolução de entidades

Não há chave comum entre as fontes. O FBref é a referência: cada linha dele é pareada com as
demais fontes (`entity_resolution.py`).

1. **Clubes.** Atribuição 1-para-1 por liga e temporada, pelo algoritmo húngaro sobre a
   similaridade dos nomes normalizados, mais *aliases* manuais (`data/external/team_aliases.csv`,
   por exemplo PSG, Wolves e Inter).
2. **Nomes.** A normalização translitera (ø, ß, ł, đ, æ, ı…), remove diacríticos e pontuação. O
   escore combina **Levenshtein** (`token_sort_ratio`) com **Jaro-Winkler**, com o máximo de
   `token_set_ratio` para apelidos parciais. Uma regra de sobrenome + inicial cobre diminutivos (Max
   ↔ Maximilian).
3. **Restrições.** Ano de nascimento com diferença de no máximo 1 ano (resolve homônimos),
   concordância de minutagem com Understat e Sofascore, e atribuição 1-para-1 dentro de cada clube.
4. **Estágios em cascata.** São aplicados nesta ordem:
   1. bloco por clube;
   2. *fallback* na liga (empréstimos no meio da temporada);
   3. propagação pelo id longitudinal do FBref;
   4. compartilhamento da linha agregada de temporada (transferidos);
   5. temporadas vizinhas;
   6. resolução de conflitos pela maioria ponderada pelos minutos;
   7. **correspondências manuais** (45 atletas em `data/external/manual_player_overrides.csv`).

### Resultado do pareamento

Universo-alvo: 15.896 linhas de jogadores de linha com ao menos 90 minutos, de 2018/19 a 2024/25.

| Fonte | Pareados |
|---|---|
| Transfermarkt | **99,94%** |
| Understat | 99,44% |
| Sofascore | 99,63% |

Métodos de pareamento com o Transfermarkt, nas 22.419 linhas:

| Método | % |
|---|---|
| Exato (após a normalização) | 93,9 |
| Difuso (Levenshtein + Jaro-Winkler) | 5,1 |
| Correspondência manual | 0,6 |
| Difuso validado por minutagem e nascimento, *fallback* na liga, temporada vizinha | 0,1 |
| Sem par | 0,2 |

Os casos manuais concentram os três tipos de ruído previstos na especificação (§5.2):
- **apelidos sem semelhança ortográfica:** *Pepelu* = José Luis García Vayá, *Simy* = Simeon
  Nwankwo, *Chicharito* = Javier Hernández;
- **sobrenomes diferentes:** Jonny Castro/Otto;
- **transliterações:** Babis/Charalampos Lykogiannis.

## 4. Valores de mercado em T, T+1 e T+2

Para cada horizonte $k$, $V_{T+k}$ é o **último valor publicado até 30 de junho do ano de término
da temporada $T+k$**, obtido por `merge_asof` retroativo. Avaliações com mais de 183 dias são
descartadas. A construção garante que $V_T$ nunca incorpore informação posterior ao fim da temporada.
O histórico do Transfermarkt foi coletado em out/2026 e cobre até 30/06/2026: o alvo de $T+1$ de
2024/25 e o de $T+2$ de 2023/24 já são observados; o de $T+2$ de 2024/25, não (`config.VALUE_CUTOFF`).

## 5. Contexto do clube em T

Variáveis do clube principal do atleta (`club_context.py`), todas conhecidas na data-âncora de $T$:

| Variável | Fonte | Definição |
|---|---|---|
| Posição final normalizada | resultados das partidas no Understat | 0 = campeão, 1 = último; ordem por pontos por jogo, saldo por jogo e gols pró por jogo |
| Saldo de gols por jogo | resultados no Understat | saldo na liga em $T$ dividido pelos jogos |
| Promovido | FBref | o clube não estava nas Big Five em $T-1$ |
| Rebaixado | FBref (Understat para 2025/26) | o clube não está nas Big Five em $T+1$ |
| Participação europeia | Sofascore (Liga dos Campeões, Liga Europa, Liga Conferência) | maior competição UEFA disputada em $T$, inclusive qualificatórias (0 a 3) |
| Elo e variação do Elo | ClubElo (opcional) | indisponível nesta coleta |

- **Cobertura:** 100% em todas as temporadas, exceto o Elo (0%).
- **Coerência:** 13 a 15 clubes promovidos e rebaixados por temporada (3 por liga, com variações por
  *play-offs* e pela redução da Ligue 1 a 18 clubes em 2023/24, que rebaixou 4 em 2022/23).
- **Não antecipação.** Classificação, promoção e rebaixamento são decididos até o fim da temporada,
  inclusive os *play-offs* de maio e junho; ler o pertencimento em $T+1$ só recupera um fato consumado
  na âncora. A **vaga europeia para $T+1$ fica de fora**, porque os *play-offs* da Liga Europa e da
  Liga Conferência são disputados em agosto, depois da âncora.
- **Limitação:** a classificação calculada não aplica punições administrativas de pontos (por exemplo,
  Everton e Nottingham Forest em 2023/24); o saldo por jogo não é afetado.

## 6. Dataset bruto

- **22.419 linhas × 180 colunas.** Por temporada: 2017/18 2.692; 2018/19 2.656; 2019/20 2.732;
  2020/21 2.822; 2021/22 2.922; 2022/23 2.889; 2023/24 2.852; 2024/25 2.854.
- **Prefixos das colunas:** `fb_` (FBref), `us_` (Understat), `ss_` (Sofascore), `tm_`
  (Transfermarkt: elenco, perfil `tm_profile_*` e valores `tm_value_{T,T1,T2}`) e `ctx_` (contexto do
  clube).
- **Rastreabilidade:** `*_match_method`, `*_match_score`, `*_is_season_aggregate`.
- **Validações:** chave (temporada, jogador, clube) única e contagens de clubes por liga coerentes.
  O dicionário de colunas está no `_schema.json`.

## 7. Pontos para a seção de Metodologia

- O FBref deixou de ser a fonte das métricas avançadas, que agora vêm do Sofascore e do Understat.
  Duelos aéreos passaram a vir do Sofascore.
- O xG do Sofascore só existe a partir de 2022/23. Por isso, o Understat é a fonte única de xG, o
  que mantém a série homogênea.
- O contrato histórico não é observável no Transfermarkt: as páginas de elenco de temporadas
  passadas não trazem o contrato vigente, e o Wayback não arquiva a visão completa. O notebook 01
  coleta os insumos (contrato atual, chegada, renovação e transferências), e o notebook 03
  reconstrói o contrato em $T$.
- O contexto do clube combina três fontes já ligadas ao painel; a vaga europeia para $T+1$ não entra
  porque parte dela se decide depois da âncora.
- Os bloqueios do Transfermarkt e do FBref e a indisponibilidade do ClubElo estão documentados na
  seção 2 e nas limitações (relatório 06).
