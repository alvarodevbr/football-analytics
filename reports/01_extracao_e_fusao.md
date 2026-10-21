# 01 — Extração e Fusão das Fontes de Dados

**Notebook:** `notebooks/01_data_extraction.ipynb` ·
**Código:** `src/football_analytics/{fbref,understat,sofascore,transfermarkt,http,browser,entity_resolution,valuation,pipeline}.py`
**Saída:** `data/raw/football_raw_dataset.parquet` (+ `.csv`, `_schema.json`) e tabelas intermediárias em `data/interim/`

[← Índice](README.md)

## 1. Objetivo e escopo

O notebook constrói o dataset bruto consolidado. Cada linha é uma observação **jogador–clube–temporada**
das Big Five, com o desempenho em $T$, atributos demográficos e contratuais e valores de mercado
em $T$, $T+1$ e $T+2$.

- **Temporadas de referência:** 2019/20 a 2023/24.
- **Temporada adicional:** 2018/19, coletada apenas para construir as defasagens de 2019/20
  (`is_lag_only`).
- **Sem filtros nesta etapa.** O corte $p_{90}$, a minutagem mínima e a exclusão de goleiros são
  validados no notebook 02 e aplicados no 03.
- **Coleta do zero.** Nenhum cache de projetos anteriores foi reutilizado. Todos os caches ficam
  em `data/raw/`.

## 2. Arquitetura de fontes

Em **20/01/2026**, a Sports Reference retirou do FBref todas as métricas avançadas licenciadas pela
Opta, inclusive das temporadas históricas. As páginas *passing*, *possession*, *gca* e *defense*
continuam publicadas, mas com as colunas vazias, como documenta o diagnóstico de preenchimento do
notebook. Para preservar o vetor tático-técnico da especificação, a arquitetura passou a ter quatro
fontes:

| Fonte | Acesso | Variáveis | Papel |
|---|---|---|---|
| **FBref** | soccerdata 1.9.1 + Chrome (Cloudflare exige navegador não *headless*) | minutos, jogos, gols, assistências, chutes, cartões, faltas, cruzamentos, interceptações, desarmes ganhos, gols pró/contra em campo, ano de nascimento | Espinha dorsal da amostra e id longitudinal |
| **Understat** | soccerdata (HTTP) | xG, npxG, xA, xGChain, xGBuildup, passes-chave | Família de gols esperados |
| **Sofascore** | API JSON por clube, consultada dentro do Chrome | passes, bolas longas, terço final, passes-chave, grandes chances, dribles, duelos aéreos e no chão, desarmes, cortes, perdas de posse, nota | Vetor tático que o FBref deixou de publicar |
| **Transfermarkt** | `requests` (HTML + `/ceapi`) | valor de mercado datado, nascimento, altura, pé, nacionalidades, posição granular, contrato atual, transferências | Base financeira, alvos e controles |

### Volume coletado

| Tabela | Linhas | Observação |
|---|---|---|
| FBref (jogador–clube–temporada, 6 temporadas) | 16.873 | 4 páginas com dados × 6 temporadas, em páginas combinadas das Big Five |
| Understat | 16.446 | Uma linha por jogador e clube; transferidos aparecem agregados |
| Sofascore | 16.462 | Consulta por clube e temporada |
| Elencos do Transfermarkt | 22.715 | 586 páginas de elenco por liga e temporada |
| Histórico de valores de mercado | 156.824 avaliações | 5.955 jogadores, endpoint `/ceapi/marketValueDevelopment` |
| Perfis (contrato atual) | 5.955 | Página `/profil/spieler/{id}` |
| Histórico de transferências | 67.606 movimentos | `/ceapi/transferHistory/list/{id}`, que exige os cabeçalhos XHR |

A coleta do Transfermarkt somou cerca de 18 mil requisições, com espera de 1 s e *jitter*. Depois
dela, o site passou a responder com um **captcha do AWS WAF** (HTTP 405 com
`x-amzn-waf-action`). O cliente HTTP trata isso como bloqueio: interrompe as requisições de rede
sem insistir e continua lendo o cache. Os 44 jogadores pareados por correspondência manual depois
do bloqueio ainda não têm dados por jogador (ver relatório 06).

## 3. Resolução de entidades

Não há chave comum entre as fontes. O FBref é a referência: cada linha dele é pareada com as
demais fontes (`entity_resolution.py`).

1. **Clubes.** Atribuição 1-para-1 por liga e temporada, pelo algoritmo húngaro sobre a
   similaridade dos nomes normalizados, mais 13 *aliases* manuais (`data/external/team_aliases.csv`,
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

Universo-alvo: 11.399 linhas de jogadores de linha com ao menos 90 minutos, de 2019/20 a 2023/24.

| Fonte | Pareados |
|---|---|
| Transfermarkt | **99,96%** |
| Understat | 99,4% |
| Sofascore | 99,5% |

Métodos de pareamento com o Transfermarkt, nas 16.873 linhas:

| Método | Linhas | % |
|---|---|---|
| Exato (após a normalização) | 15.857 | 94,0 |
| Difuso (Levenshtein + Jaro-Winkler) | 859 | 5,1 |
| Correspondência manual | 114 | 0,7 |
| Difuso validado por minutagem e nascimento | 9 | 0,05 |
| *Fallback* na liga | 2 | 0,01 |
| Sem par | 32 | 0,2 |

Os casos manuais concentram os três tipos de ruído previstos na especificação (§5.2):
- **apelidos sem semelhança ortográfica:** *Pepelu* = José Luis García Vayá, *Simy* = Simeon
  Nwankwo, *Chicharito* = Javier Hernández;
- **sobrenomes diferentes:** Jonny Castro/Otto;
- **transliterações:** Babis/Charalampos Lykogiannis.

## 4. Valores de mercado em T, T+1 e T+2

Para cada horizonte $k$, $V_{T+k}$ é o **último valor publicado até 30 de junho do ano de término
da temporada $T+k$**, obtido por `merge_asof` retroativo. Avaliações com mais de 183 dias são
descartadas. A construção garante que $V_T$ nunca incorpore informação posterior ao fim da
temporada. O histórico do Transfermarkt cobre $V_{T+2}$ até 30/06/2026, já observado.

## 5. Dataset bruto

- **16.873 linhas × 171 colunas.** Por temporada: 2018/19 2.656; 2019/20 2.732; 2020/21 2.822;
  2021/22 2.922; 2022/23 2.889; 2023/24 2.852.
- **Prefixos das colunas:** `fb_` (FBref), `us_` (Understat), `ss_` (Sofascore) e `tm_`
  (Transfermarkt: elenco, perfil `tm_profile_*` e valores `tm_value_{T,T1,T2}`).
- **Rastreabilidade:** `*_match_method`, `*_match_score`, `*_is_season_aggregate`.
- **Validações:** chave (temporada, jogador, clube) única e contagens de clubes por liga coerentes.
  O dicionário de colunas está no `_schema.json`.

## 6. Pontos para a seção de Metodologia

- O FBref deixou de ser a fonte das métricas avançadas, que agora vêm do Sofascore e do Understat.
  Duelos aéreos passaram a vir do Sofascore.
- O xG do Sofascore só existe a partir de 2022/23. Por isso, o Understat é a fonte única de xG, o
  que mantém a série homogênea.
- O contrato histórico não é observável no Transfermarkt: as páginas de elenco de temporadas
  passadas não trazem o contrato vigente, e o Wayback não arquiva a visão completa. O notebook 01
  coleta os insumos (contrato atual, chegada, renovação e transferências), e o notebook 03
  reconstrói o contrato em $T$.
- O Sofascore inclui na lista de times de algumas temporadas clubes dos *playoffs* de acesso, com
  poucos minutos (Holstein Kiel, Rodez, Paris FC, Troyes). Eles ficam sem mapeamento e não afetam
  o dataset.
