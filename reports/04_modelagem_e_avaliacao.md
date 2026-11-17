# 04 — Modelagem e avaliação (versão 4)

Notebook: `notebooks/04_model_training.ipynb`. Tabelas em `tables/04_model_training/` e figuras em
`figures/04_model_training/`. O relatório da v3 está em `archive/v3_2026-10-23/reports/`.

## 1. Execução

| Etapa | Detalhe |
|---|---|
| Desenvolvimento, teste oculto | 18 tarefas da pré-etapa do alvo, 142 tarefas otimizadas e 108 com hiperparâmetros fixos; cerca de 100 minutos a frio, 649 CPU-min (`tab01`) |
| Protocolo | `reports/protocol_v4.json`, *commit* `8883da8` (17/11/2026) |
| Revelação | `FA_REVEAL_HOLDOUT=1`: o notebook confirmou que o protocolo estava *commitado* e idêntico ao reproduzido (*commit* `8883da8`) antes de calcular métricas do teste |

## 2. Escolhas no desenvolvimento

**Alvo (seção 1.1, `tab00_escolha_do_alvo`).** RMSE média das dobras de desenvolvimento, *ensemble* com
hiperparâmetros-padrão:

| Horizonte | Nominal | Excesso, *drift* médio | Excesso, *drift* por regime |
|---|---|---|---|
| $T+1$ | **0,5146** | 0,5148 | 0,5372 |
| $T+2$ | **0,7231** | 0,7235 | 0,8108 |

O excesso ao *drift* empata com o nominal: a revisão de abril de 2020 não distorce a ordenação aprendida.
Prever o *drift* da temporada pelo regime de $T$ piora muito o nível.

**Arquitetura (seção 4.1, `tab07_decisao_arquitetura`).**
- $T+1$: o *ensemble* com interações restritas passa a regra (RMSE de validação de 0,5171 contra 0,5185;
  $p_{\text{Holm}} < 0{,}001$).
- $T+2$: nada passa.
  - O *pooling* 3 × 3, escolhido na v3, cai para $p_{\text{Holm}} = 0{,}085$ com o erro-padrão de dois
    agrupamentos.
  - Os modelos segmentados erram mais (+0,010 a +0,022).
- κ do *pooling*: 3.200 ($T+1$) e 1.600 ($T+2$).

**Camada de magnitude e probabilidade de decisão (seção 4.2).** As tabelas `tab12c` e `tab15`, e o
relatório 06 (§2).

## 3. Teste final

**Arquitetura final** (`tab05`, ICs em `tables/05_model_interpretation/tab00_teste_final_arquitetura_final_ic95`):

| Métrica | $T+1$ (2024/25): *ensemble* com interações | $T+2$ (2023/24): *ensemble* global | v3 pré-registrada |
|---|---|---|---|
| RMSE do log-retorno | 0,540 [0,501; 0,581] | 0,722 [0,689; 0,756] | 0,543 / 0,728 |
| $R^2$ do log-retorno | 0,337 [0,304; 0,369] | 0,382 [0,341; 0,416] | 0,329 / 0,371 |
| $\rho$ de Spearman | 0,570 [0,536; 0,601] | 0,608 [0,570; 0,643] | 0,560 / 0,604 |
| Decil superior que se valorizou ($\Delta V\% > 0$) | 71% (média 27,5%) | 64% (média 26%) | 71% / 66% |
| Redução do RMSE frente a "sem variação" | −20% | −27% | −19% / −27% |

**Comparações (`tab06`, `tab06b`).**
- Ridge, CatBoost e os dois *ensembles* empatam em $T+1$.
- O conjunto de confiança contém o Ridge e os dois *ensembles* nos dois horizontes.
- Os segmentados CatBoost e o OLS ficam fora.

**H1 (`tab08`, `fig08`).**
- Sem desempenho (T, T−1) e valor justo:
  - $T+1$: RMSE +0,011 ($p_{\text{Holm}} < 0{,}001$);
  - $T+2$: +0,009 ($p_{\text{Holm}} = 0{,}069$).
- Caminho aninhado em $T+1$: $R^2$ de 0,264 (ciclo de vida, contexto e lesões) → 0,300 (+ nível de preço)
  → 0,308 (+ trajetória) → 0,335 (+ desempenho e valor justo).
- Retirar a trajetória do valor não muda o modelo completo.

**Empilhamento (seção 7.1, `tab08b`, desenvolvimento).**

| Membro | RMSE com 1º estágio | sem 1º estágio | Perda |
|---|---|---|---|
| CatBoost ($T+1$) | 0,5215 | 0,5282 | +0,0067 |
| XGBoost ($T+1$) | 0,5211 | 0,5254 | +0,0043 |
| Ridge ($T+1$) | 0,5226 | 0,5239 | +0,0013 |

Sem os gaps, o Ridge supera as árvores: o empate da versão completa não é um artefato do empilhamento.

**H2 (`tab09`, `fig03`).** Os modelos por célula erram mais em todas as matrizes; o *pooling* empata.

**Magnitude (`tab12`, `tab12d`, `fig06`).** Erro absoluto entre $\Delta V\%$ médio previsto e observado
por decil, no teste:

| Camada | $T+1$: média / 10º decil | $T+2$: média / 10º decil |
|---|---|---|
| v3 (Duan) | 21,6 / 181,3 | 25,9 / 191,7 |
| Duan (v4) | 23,3 / 186,7 | 26,4 / 179,0 |
| Heterocedástica | 17,3 / 148,6 | **17,9 / 120,9** |
| Forma multiquantil | **14,8 / 133,6** | 15,5 / 114,1 |
| Integração multiquantil | 25,7 / 202,7 | 29,4 / 245,7 |
| Duas partes | 19,9 / 169,6 | 25,9 / 213,0 |

Em $T+1$, o 10º decil é previsto em +189% contra +323% observados (v3: +126% contra +307%).

**Intervalos (`tab13`).** Os intervalos conformalizados de 80% cobrem 82,7% ($T+1$) e 82,6% ($T+2$) do
teste, com largura mediana de 105 e 138 p.p.

**Probabilidade de decisão (`tab15b`).** Brier no teste:

| Escore | $T+1$ | $T+2$ |
|---|---|---|
| Regressão | 0,132 | 0,137 |
| Média dos classificadores | 0,129 | 0,131 |
| Taxa-base | 0,171 | 0,173 |

**Robustez (`tab14`).**
- Hiperparâmetros-padrão, taxas cruas, alvo anualizado e retirada da nota do Sofascore não mudam o
  resultado.
- Retirar 2018/19 piora $T+2$ (0,732).
- O contrato reconstruído (*look-ahead*) não melhora o XGBoost de forma significativa.
