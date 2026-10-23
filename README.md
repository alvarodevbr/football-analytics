# Football Transfer Risk & AI Scouting

Previsão da variação do valor de mercado ($\Delta V$, $\Delta V\%$) em $T+1$/$T+2$ para jogadores de linha
das Big Five (2018/19–2024/25; 2017/18 só para defasagens). O contexto metodológico está em `CLAUDE.md` e
`specs/`.

**Versão 3 (out/2026).** Teste final de $T+1$ em 2024/25, pré-registrado em `reports/protocol_v3.json`:
o notebook 04 só revela as métricas desse teste depois que o protocolo é *commitado*. A versão 2 está
arquivada em `reports/archive/v2_2026-10-19/` (e completa no commit `dc93ba2`).

## Ambiente

```bash
python3 -m venv --without-pip .venv            # sem python3-venv no sistema; com ele, omita --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/sbase get cft                         # Chrome for Testing (FBref/Sofascore exigem navegador)
.venv/bin/python -m ipykernel install --user --name football-analytics --display-name "football-analytics (.venv)"
```

## Notebook 01 — extração

`notebooks/01_data_extraction.ipynb` (kernel `football-analytics (.venv)`):

- `SMOKE_TEST = True` (ou `FA_SMOKE_TEST=1`) roda um recorte mínimo em `data/smoke_test/`.
- Coleta completa: várias horas a frio, a maior parte no Transfermarkt. Tudo fica em cache em
  `data/raw/`, então basta reexecutar para retomar. `MAX_FETCHES_PER_RUN` divide a coleta em sessões.
- O FBref e o Sofascore abrem uma janela do Chrome. O modo headless é bloqueado pelos anti-bots. Sem o
  pacote de sistema `python3-tk`, o desafio do Cloudflare é contornado reabrindo a página com o driver
  desconectado (`fbref._reconnect_through_challenge`), sem clique.
- O contexto do clube (classificação, promoção, rebaixamento, participação europeia) vem do
  Transfermarkt; o Elo do ClubElo é opcional e sai do vetor se a API estiver fora do ar.

Execução não interativa:

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 01_data_extraction.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=football-analytics
```

Saída: `data/raw/football_raw_dataset.parquet` (+ `.csv`, `_schema.json`).

Se o Transfermarkt responder com captcha (HTTP 405 + `x-amzn-waf-action`), a coleta de rede é
suspensa na hora e o cache continua sendo lido. Basta reexecutar mais tarde. A coleta roda em sessões de
até 1.500 requisições por endpoint (`MAX_FETCHES_PER_RUN`).

## Notebook 02 — EDA e verificação da amostra

`notebooks/02_eda.ipynb` lê o dataset bruto, aplica as regras de amostra de `sample.py` e as audita.
As mesmas regras serão aplicadas no notebook 03. O notebook exporta:

- `reports/figures/02_eda/*.png`: 300 dpi, 16 cm de largura, sem título embutido (ABNT). As
  legendas sugeridas e a fonte de cada figura ficam em `figures_index.md`.
- `reports/tables/02_eda/*.csv`: UTF-8 com BOM, para abrir direto no Excel ou Word.

## Notebook 03 — pré-processamento

`notebooks/03_preprocessing.ipynb` gera o painel de modelagem `data/processed/train_test_lags.parquet`.
Cada linha é um jogador–temporada, com as features em T e os alvos em T+1/T+2. O notebook também
grava `data/processed/feature_sets.json`, com as listas de features, os alvos, as variáveis de
sensibilidade (*look-ahead*) e as dobras da janela expandida que o notebook 04 vai consumir. As taxas
por 90 minutos passam por encolhimento empírico-bayesiano por temporada × posição (as versões cruas ficam
como `*_raw`). A imputação e a padronização ficam no notebook 04, dentro de cada dobra. Os alvos de
$T+1$ de 2024/25 ficam ocultos nos diagnósticos dos notebooks 02 e 03.

## Notebook 04 — treinamento

`notebooks/04_model_training.ipynb` treina, nos horizontes T+1 e T+2:

- **Modelos globais:** mediana ingênua, OLS, Ridge, Random Forest, XGBoost, CatBoost e o *ensemble*
  (média de CatBoost, XGBoost e Ridge).
- **Segmentos:** mediana, Ridge e CatBoost por categoria nas matrizes 4 × 3 (C1–C12) e 3 × 3 de
  Khalife et al. (K1–K9), mais o *pooling* parcial entre segmento e modelo global.
- **Ablação e robustez:** o *ensemble* sem cada bloco de variáveis, num caminho aninhado, com
  hiperparâmetros-padrão, com taxas cruas e sem 2018/19 no treino.

A otimização bayesiana (`skopt.gp_minimize`) usa as três últimas dobras de validação; XGBoost e
CatBoost têm taxa de aprendizado fixa e número de árvores por parada antecipada temporal. A inferência
usa *bootstrap* por clube × temporada, teste pareado de perdas com correção de Holm e o conjunto de
confiança de modelos (MCS).

Execução em duas etapas:

```bash
cd notebooks
../.venv/bin/jupyter nbconvert --to notebook --execute 04_model_training.ipynb --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=football-analytics
git add ../reports/protocol_v3.json && git commit -m "Pré-registro do protocolo v3"
FA_REVEAL_HOLDOUT=1 ../.venv/bin/jupyter nbconvert --to notebook --execute 04_model_training.ipynb --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=football-analytics
```

Cada tarefa vai para o cache em `models/04_model_training/cache/` (a chave inclui o hash do painel),
então a segunda etapa reaproveita tudo. Com `FA_SMOKE_TEST=1`, o notebook roda com 15% das linhas e grava
em `data/smoke_test/04/`.

Saídas:
- modelos finais e `manifest.json` em `models/04_model_training/final/`;
- `data/processed/predictions_oof.parquet` e `predictions_best.parquet`.

## Notebook 05 — interpretação e simulação

`notebooks/05_model_interpretation.ipynb` calcula os valores SHAP do *ensemble* final: TreeExplainer
para o CatBoost e o XGBoost e forma fechada para o Ridge, com a aditividade verificada. O notebook
analisa os determinantes da valorização no total, nas 12 categorias (com ICs por *bootstrap*) e nas
posições, mede a estabilidade das explicações com os modelos de cada dobra, testa a premissa *buy-low*
pelo gap de avaliação e simula estratégias *buy-low / sell-high* em *backtest* fora da amostra, com
custos de transação, intenção de tratar (atletas que saem da base), teste do sinal entre temporadas e
risco medido por VaR e CVaR. Termina com uma lista prospectiva 2024/25 → 30/06/2027, datada e versionada.
Exige o teste final revelado no notebook 04.

Saídas:
- `data/processed/shap_values_T{1,2}.parquet`;
- `reports/figures/05_model_interpretation/` e `reports/tables/05_model_interpretation/`.

## Código e testes

| Módulo (`src/football_analytics/`) | Responsabilidade |
|---|---|
| `config.py` | ligas, temporadas, caminhos, parâmetros de coleta |
| `fbref.py` | fetch via soccerdata (Chrome) + parser próprio por `data-stat` |
| `understat.py` | xG, npxG, xA, xGChain, xGBuildup |
| `sofascore.py` / `browser.py` | estatísticas táticas por clube via API JSON no Chrome (UC) |
| `transfermarkt.py` / `http.py` | elencos, valores, perfil/contrato, transferências |
| `entity_resolution.py` | mapeamento de clubes e jogadores entre as fontes |
| `club_context.py` | contexto do clube em T: classificação, promoção/rebaixamento, Europa, ClubElo (opcional) |
| `valuation.py` | valor vigente na âncora (30/jun de T+k), sem *look-ahead* |
| `sample.py` | unidade jogador–temporada, posição, idade, 12 categorias, alvos ΔV/ΔV%, funil p90 |
| `features.py` | agregação jogador–temporada, taxas /90 com encolhimento empírico-bayesiano, demografia, clube, defasagens |
| `contracts.py` | contrato vigente em T reconstruído (observado/estimado, Bosman) |
| `feature_catalog.py` | dicionário de variáveis: papel, grupo, fonte, descrição |
| `validation.py` | dobras da janela expandida (purga em T+2) |
| `modeling.py` | pré-processamento na dobra, parada antecipada temporal, otimização GP, métricas, *bootstrap* por cluster, teste pareado, Holm, MCS, cache |
| `interpretation.py` | SHAP do ensemble e das dobras, estabilidade, carteiras com custos e intenção de tratar, VaR/CVaR |
| `plotting.py` | estilo das figuras do artigo (ABNT, paleta acessível), números no padrão brasileiro |
| `pipeline.py` | etapas do notebook 01 |

```bash
.venv/bin/pytest
```

As tabelas manuais da resolução de entidades ficam em `data/external/` (ver o README da pasta).
