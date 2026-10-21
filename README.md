# Football Transfer Risk & AI Scouting

Previsão da variação do valor de mercado ($\Delta V$, $\Delta V\%$) em $T+1$/$T+2$ para jogadores de linha
das Big Five (2019/20–2023/24). O contexto metodológico está em `CLAUDE.md` e `specs/`.

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
- O FBref e o Sofascore abrem uma janela do Chrome. O modo headless é bloqueado pelos anti-bots.

Execução não interativa:

```bash
cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute 01_data_extraction.ipynb \
    --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=football-analytics
```

Saída: `data/raw/football_raw_dataset.parquet` (+ `.csv`, `_schema.json`).

Se o Transfermarkt responder com captcha (HTTP 405 + `x-amzn-waf-action`), a coleta de rede é
suspensa na hora e o cache continua sendo lido. Basta reexecutar mais tarde.

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
sensibilidade (*look-ahead*) e as dobras da janela expandida que o notebook 04 vai consumir. A
imputação e a padronização ficam no notebook 04, dentro de cada dobra.

## Notebook 04 — treinamento

`notebooks/04_model_training.ipynb` treina 6 algoritmos em 2 arquiteturas e 2 horizontes, num total de
156 tarefas:

- **Algoritmos:** mediana ingênua, OLS, Ridge, Random Forest, XGBoost e CatBoost.
- **Arquiteturas:** um modelo global e 12 modelos segmentados, um por categoria.
- **Horizontes:** T+1 e T+2.

A otimização bayesiana usa processo gaussiano (`skopt.gp_minimize`) nas dobras de validação da
janela expandida. A última dobra (2023/24) fica reservada como teste final.

Cada tarefa vai para o cache em `models/04_model_training/cache/`, então reexecutar retoma de onde
parou. A primeira execução leva cerca de 2 horas num processador de 4 núcleos. Com
`FA_SMOKE_TEST=1`, o notebook roda com 15% das linhas e grava em `data/smoke_test/04/`.

Saídas:
- modelos finais e `manifest.json` em `models/04_model_training/final/`;
- `data/processed/predictions_oof.parquet` e `predictions_best.parquet`.

## Notebook 05 — interpretação e simulação

`notebooks/05_model_interpretation.ipynb` calcula os valores SHAP do *ensemble* final: TreeExplainer
para o CatBoost e o XGBoost e forma fechada para o Ridge, com a aditividade verificada. O notebook
analisa os determinantes da valorização no total, nas 12 categorias e nas posições, gera gráficos
de dependência e explicações individuais, e simula estratégias *buy-low / sell-high* em *backtest*
fora da amostra, com o risco medido por VaR e CVaR.

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
| `valuation.py` | valor vigente na âncora (30/jun de T+k), sem *look-ahead* |
| `sample.py` | unidade jogador–temporada, posição, idade, 12 categorias, alvos ΔV/ΔV%, funil p90 |
| `features.py` | agregação jogador–temporada, taxas /90, demografia, clube, defasagens |
| `contracts.py` | contrato vigente em T reconstruído (observado/estimado, Bosman) |
| `feature_catalog.py` | dicionário de variáveis: papel, grupo, fonte, descrição |
| `validation.py` | dobras da janela expandida (purga em T+2) |
| `modeling.py` | pré-processamento na dobra, algoritmos, otimização GP, métricas, cache |
| `interpretation.py` | SHAP do ensemble, agregação, carteiras, VaR/CVaR |
| `plotting.py` | estilo das figuras do artigo (ABNT, paleta acessível), números no padrão brasileiro |
| `pipeline.py` | etapas do notebook 01 |

```bash
.venv/bin/pytest
```

As tabelas manuais da resolução de entidades ficam em `data/external/` (ver o README da pasta).
