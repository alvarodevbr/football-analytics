# 05 — Interpretação e simulação (versão 4)

Notebook: `notebooks/05_model_interpretation.ipynb`. Tabelas em `tables/05_model_interpretation/` e
figuras em `figures/05_model_interpretation/`. O relatório da v3 está em `archive/v3_2026-10-23/reports/`.

## 1. Modelo explicado

- **Componentes:**
  - $T+1$: *ensemble* com interações restritas (CatBoost, XGBoost com `interaction_constraints` e Ridge
    com coeficientes por faixa × posição);
  - $T+2$: *ensemble* global.
- **SHAP:** `interpretation.ensemble_shap`.
  - TreeExplainer nas árvores e forma fechada no Ridge.
  - As interações do Ridge são somadas à variável original, e a aditividade é verificada.
  - Idade, idade² e faixa etária são somadas numa coalizão (`interpretation.coalition_shap`).
- **Teste final da arquitetura final**, com ICs (`tab00_teste_final_arquitetura_final_ic95`):

| Horizonte | $R^2$ | $\rho$ |
|---|---|---|
| $T+1$ | 0,337 [0,304; 0,369] | 0,570 [0,536; 0,601] |
| $T+2$ | 0,382 [0,341; 0,416] | 0,608 [0,570; 0,643] |

## 2. Determinantes (`tab01`, `tab01b`, `fig01`, `fig02`)

- **Coalizão etária.** É a maior contribuição: média de $|\phi|$ de 0,21 em $T+1$ e 0,38 em $T+2$.
- **Gap de avaliação e nível de preço.** Seguem o gap (0,050 e 0,070), $\ln V_T$ (0,041 e 0,072), a
  variação do valor em 6 meses, a variação da participação nos minutos e a nota do Sofascore.
- **Grupos ($T+1$ / $T+2$).**

| Grupo | $T+1$ | $T+2$ |
|---|---|---|
| Desempenho | 26,5% | 23,6% |
| Demografia | 26,3% | 34,3% |
| Contexto | 10,5% | 7,4% |
| Defasagem | 9,5% | 8,8% |
| Mercado | 8,4% | 9,9% |
| Valor justo | 7,6% | 7,2% |
| Dinâmica de mercado | 7,3% | 6,1% |
| Disponibilidade física | 3,8% | 2,8% |

- **Sem a nota do Sofascore.** O crédito passa a grandes chances perdidas, % de jogos completos, duelos
  aéreos e finalizações.
- **Figura 1.** Rótulo $\ln V_T$, convenção de cores do shap (vermelho = alto) e coalizão etária.

## 3. Carreira e H5 (`tab02`, `tab02b`, `tab02c`, `fig03`, `fig03b`)

**Participação em $T+1$** (Figura 2, agora com rótulos "DEF 16–23" etc. e os dois horizontes no título):

| Faixa | Desempenho | Demografia |
|---|---|---|
| 24–28 | 29–35% | 14–16% |
| 33+ | 19–27% | 36–40% |

**H5 sustentada.** A diferença auge − veteranos é de +9,6 p.p. [9,2; 10,0] em $T+1$ e +9,5 p.p.
[9,1; 9,9] em $T+2$.

**Estabilidade entre dobras.** A diferença fica entre 7,4 e 9,1 p.p. e entre 7,7 e 9,6 p.p. O $\tau$ de
Kendall médio é de 0,59 e 0,64.

## 4. H3 (`tab11`, `tab11b`)

**Quintis extremos do gap, no teste:**

| Horizonte | Mais subavaliado | Mais sobreavaliado | ICs |
|---|---|---|---|
| $T+1$ | −10,0% [−14,3; 0,0] | −20,0% [−25,0; −16,7] | disjuntos |
| $T+2$ | −25,0% | −40,0% | tocam-se em −33,3% |

**Correlação parcial.** −0,20 e −0,13. Com o gap só de desempenho, −0,19 e −0,11.

## 5. Carteiras e H4 (`tab06`, `tab07`, `tab07b`, `tab07c`, `fig08`)

**Cenário principal da v4.** Custos de 10% + 10%; aposentadoria ou falta de clube valem zero, avaliação
antiga vale o último valor. ROI médio das temporadas, todos os candidatos:

| Estratégia | $T+1$ | $T+2$ |
|---|---|---|
| Modelo | **58,1%** | **96,7%** |
| Modelo prudente (P10) | 29,4% | 65,2% |
| 10% mais jovens | 25,9% | 55,5% |
| Jovens titulares | −1,2% | 5,0% |
| *Momentum* | 9,2% | 15,5% |
| Jovens 16–23 ao acaso | 6,5% | 21,8% |
| Aleatória | −19,1% | −24,3% |

- **Teste do sinal.** O modelo vence o acaso em 6 de 6 temporadas ($p$ = 0,016) e 4 de 4 ($p$ = 0,06).
  Supera as heurísticas e o acaso em todas as temporadas, universos e cenários.
- **Única exceção.** Em três temporadas do cenário de preço observado na faixa de € 2–15 mi, a variante
  "prudente" do próprio modelo rende mais que ele.
- **Alfa sobre os 10% mais jovens:** +32 p.p. e +41 p.p.

| Cenário ou universo | $T+1$ | $T+2$ |
|---|---|---|
| Universo comprável (sem os 10 elencos mais valiosos) | 61,9% | 104,4% |
| Faixa de € 2–15 mi | 41,0% | 70,2% |
| Carregamento (5% a.a. + 10% de $V_T$ a.a.) | 38,5% | 53,1% |
| Preço de aquisição observado | 45,2% | 77,7% |
| Média simples (não ponderada) dos retornos | 136% | 219% |

**Preço de aquisição observado.** A mediana de taxa/$V_T$ nas vendas definitivas é 0,86. Por grupo:
- jovens: 1,11 fora da elite e 1,45 nos elencos de elite;
- 24–28: 0,82 fora da elite;
- 29–32: 0,58 fora da elite;
- 33+: 0,63 fora da elite.

**Teste de 2024/25.** ROI de 108% contra −16% ao acaso.

**Concentração.** Sem os 3 maiores acertos, o ROI sem custos fica acima de 37% em todas as temporadas de
$T+1$.

## 6. Orçamento e risco (`tab08`, `tab08b`, `fig09`)

**Mochila por programação inteira** (`interpretation.budget_milp`): € 40 mi já com o custo de compra, até
6 atletas de € 2–15 mi, no máximo 2 por posição e 3 por faixa.
- $T+1$:
  - ROI de −20% a +127% entre as temporadas;
  - 56% de média no desenvolvimento, com P(perda) de 29%;
  - 125% no teste.
- $T+2$: 115% no desenvolvimento e 197% no teste.
- As carteiras aleatórias perdem em 74–87% dos sorteios.

**Penalidade de risco** (λ = 1, escolhido no desenvolvimento).
- Ela concentra a carteira.
- Melhora o desenvolvimento (65% e 176%) e o teste de $T+1$ (220%, 4 atletas).
- No teste de $T+2$, perde 33% (3 atletas).
- Não é robusta.

## 7. Regra de compra (métricas pré-registradas na v4; `tab13`–`tab15c`, `fig12`)

**Taxa-base de pagar a operação.** 21,7% no teste de $T+1$ e 21,4% no de $T+2$.

| Métrica (teste) | $T+1$ | $T+2$ |
|---|---|---|
| AUC do modelo | 0,824 [0,804; 0,843] | 0,831 [0,810; 0,852] |
| AUC dos 10% mais jovens | 0,802 | 0,835 |
| Modelo − mais jovens | +0,022 [0,011; 0,035] | −0,004 [−0,018; 0,008] |
| Decil superior que pagou: modelo | 65,5% | 56,9% |
| Decil superior que pagou: escore do protocolo | 65,5% | 63,1% |
| Decil superior que pagou: 10% mais jovens | 55,8% | 63,1% |
| *Brier skill* da probabilidade calibrada | 0,23 | 0,25 |

- **Probabilidade acima de 60% ($T+1$).** 71% de acerto.
- **Limiar com custo de carregamento** (39,4% e 57,0%). 61% do decil superior de $T+1$ o superam, contra
  uma taxa-base de 18%.
- **Figura 3.** Hastes de IC de Wilson e histograma das probabilidades.

## 8. Lista de observação e lista prospectiva

**Lista de observação** (`tab09`, `fig10`): 15 maiores oportunidades de € 2–15 mi em 30/06/2025.
- 73% se valorizaram; ROI de 252% sem custos e 188% no cenário principal; 80% dentro do intervalo de 80%.
- Nove são de elencos de elite.

**Lista prospectiva** (`tab12_lista_prospectiva_2425_T2`): 2024/25 → 30/06/2027.
- Emitida com os modelos de `8883da8`.
- As camadas de magnitude e conformal são reaplicadas como uma dobra seguinte, com o histórico das
  previsões fora da amostra.

## 9. Apêndices do artigo

**Sensibilidade ao teto de preço** (`tab16`, `fig13`):

| Teto de $V_T$ | AUC no teste ($T+1$) |
|---|---|
| € 5 mi | 0,86 |
| p90 | 0,82 |

O ROI do decil superior é maior entre os mais baratos.

**Coerência entre horizontes** (`tab17`): correlação entre $\hat y_1$ e $\hat y_2$ de 0,95–0,98; mesmo
sinal em 88–95% dos casos.
