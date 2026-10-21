# 05 — Interpretação (XAI) e Simulação *Buy-Low / Sell-High*

**Notebook:** `notebooks/05_model_interpretation.ipynb` · **Código:** `src/football_analytics/interpretation.py`
**Saídas:** `data/processed/shap_values_T{1,2}.parquet`, 10 figuras em
`figures/05_model_interpretation/` e 9 tabelas em `tables/05_model_interpretation/`

[← Índice](README.md)

## 1. Método

- **Modelo explicado:** o *ensemble* global final (média de CatBoost, XGBoost e Ridge), ajustado em
  todas as temporadas com alvo.
- **SHAP por componente:**
  - **CatBoost e XGBoost:** `shap.TreeExplainer`;
  - **Ridge:** forma fechada, $\phi_j = w_j(x'_j-\bar x'_j)$.

  Os SHAP do *ensemble* são a média dos três, o que preserva a aditividade
  $\hat y_i = \phi_0 + \sum_j\phi_{ij}$; ela é verificada numericamente.
- **Variáveis categóricas:** as *dummies* de liga, faixa etária e posição são somadas de volta.
- **Leitura das unidades:** em log-retorno, $\phi = +0{,}10$ equivale a cerca de +10,5% de variação
  no valor.
- **Leitura "por categoria":** agrega as explicações do modelo global dentro de cada categoria
  C1–C12. Não são 12 modelos distintos, porque o notebook 04 mostrou que o global é superior.

**Valor-base** ($\phi_0$): −0,159 em $T+1$ e −0,381 em $T+2$, ou seja, −15% e −32% de variação
mediana. A desvalorização típica da amostra é o ponto de partida de toda previsão.

## 2. Determinantes globais

![Beeswarm T+1](figures/05_model_interpretation/fig01_shap_beeswarm_T1.png)

| Bloco | Participação no $\vert\phi\vert$ total ($T+1$ / $T+2$) | Destaques |
|---|---|---|
| Demográfico | 27% / 35% | idade, idade², faixa etária |
| Desempenho tático-técnico | 32% / 24% | % dos minutos possíveis, nota Sofascore, xG/90, finalizações |
| Trajetória do valor | 20% / 24% | gap de avaliação, valor justo, variação em 6 meses, revisões, *momentum* |
| Mercado | 14% / 11% | $\log V_T$ (efeito negativo), % do valor do elenco, *ranking* do elenco |
| Contexto | 7% / 5% | pontos/jogo da equipe, liga, regime de mercado |

As variáveis mais importantes em $T+1$, em ordem, são: idade, $\log V_T$, idade², **gap de
avaliação**, variação do valor em 6 meses, pontos/jogo da equipe, % do valor do elenco e variação
da participação nos minutos. Em $T+2$, o gap de avaliação sobe para 3º lugar.

![Beeswarm T+2](figures/05_model_interpretation/fig02_shap_beeswarm_T2.png)

## 3. Determinantes ao longo da carreira (12 categorias)

![Grupos por categoria](figures/05_model_interpretation/fig03_grupos_por_categoria.png)

Participação em $T+1$, em % do $\vert\phi\vert$ total de cada categoria:

| Faixa | Demografia | Desempenho | Trajetória | Mercado |
|---|---|---|---|---|
| 16–23 (C1–C3) | 29–31 | 28–33 | 19–21 | 12–14 |
| 24–28 (C4–C6) | **15–17** | **35–41** | **22–24** | 14–15 |
| 29–32 (C7–C9) | 30–33 | 29–35 | 18–19 | 11–12 |
| 33+ (C10–C12) | **37–40** | 25–32 | 15–16 | 11–15 |

- **No auge** (24–28), o desempenho e a trajetória do valor têm o maior peso.
- **Nos veteranos**, a idade domina e o desempenho pesa menos.
- **Por posição**, os atacantes dependem mais das métricas ofensivas em todas as faixas (13–17%,
  contra 8–11%).
- **Em $T+2$**, os pesos se deslocam para idade, gap e trajetória (desempenho: 17–32%).

**Confronto com a especificação.** A especificação, apoiada em Khalife et al. (2025), previa que o
rendimento operacional governasse a valoração dos veteranos. Os dados indicam o contrário: para os
atletas de 33 anos ou mais, a idade explica cerca de 40% da variação prevista. O mercado premia o
desempenho sobretudo enquanto ele sinaliza trajetória futura.

![Top 5 por categoria](figures/05_model_interpretation/fig04_top5_por_categoria.png)

A tabela `tab03_top3_shap_por_categoria.csv`, equivalente da coluna "SHAP Top 3" da Tabela 6.1 da
especificação, mostra que idade, idade² e $\log V_T$ lideram todas as categorias. O gap de
avaliação aparece em 3º lugar entre os 24–28 anos ($T+1$) e em todas as faixas até 32 anos ($T+2$).

## 4. Variáveis tático-técnicas mais determinantes

![Desempenho por posição](figures/05_model_interpretation/fig05_desempenho_por_posicao.png)

**Nas três posições,** lideram a **participação e a regularidade**: % dos minutos possíveis,
partidas, % de jogos completos e nota média do Sofascore, todas com efeito (+).

**Por posição:**

| Posição | Métricas de destaque |
|---|---|
| Atacante | xG/90, npxG/90, gols − npxG/90 (finalização acima do esperado), finalizações/90 |
| Meio-campista | dribles certos/90, xA/90, finalizações/90; "vezes driblado" com efeito (−) |
| Defensor | finalizações e dribles: o mercado premia participação ofensiva |

Os efeitos (−) de xG + xA/90 e de minutos absolutos são **condicionais** a variáveis correlacionadas
(xG/90, % dos minutos), e não devem ser lidos isoladamente.

## 5. Dependências

![Dependência](figures/05_model_interpretation/fig06_dependencia_shap.png)

- **Idade:** efeito praticamente linear, de cerca de +0,25 aos 18 anos a −0,25 aos 35, com
  cruzamento do zero por volta dos 27.
- **Gap de avaliação:** efeito monotônico e assimétrico. Os mais subavaliados recebem até +0,15 a
  +0,20 (≈ +16% a +22%), e os sobreavaliados, no máximo −0,10. A correção é mais forte para quem
  está barato.
- **% dos minutos possíveis e nota:** relações lineares positivas. Entre jogar 10% e 90% dos
  minutos, a diferença é de cerca de 7 p.p. de valorização.
- ***Momentum* anual:** efeito pequeno e com reversão após saltos extremos. A variação de seis meses
  carrega a continuidade.
- **Distância ao pico:** atletas no pico recebem efeito positivo. Os que já estão em declínio não
  recebem expectativa de recuperação.

## 6. Explicações individuais

![Waterfall](figures/05_model_interpretation/fig07_waterfall_exemplos.png)

Atleta de maior valorização esperada em cada faixa, na previsão fora da amostra de 30/06/2024:

| Atleta | Faixa | $V_T$ | Previsto (mediana / esperança) | IC 80% | Realizado |
|---|---|---|---|---|---|
| Yoel Lago (Celta) | 16–23 | € 0,2 mi | +168% / +268% | −38% a +785% | **+400%** |
| Antonios Papadopoulos (Dortmund) | 24–28 | € 1,0 mi | +47% / +61% | −37% a +208% | **+250%** |
| Emil Krafth (Newcastle) | 29–32 | € 2,5 mi | −14% / −7% | −54% a +31% | **−20%** |
| Cristian Herrera (Las Palmas) | 33+ | € 0,3 mi | −17% / −13% | −61% a +13% | **−33%** |

Nas faixas de 29 anos ou mais, até o melhor caso tem previsão negativa: a idade domina as
contribuições positivas.

## 7. Simulação de recrutamento (*backtest* fora da amostra)

Em cada temporada de teste, a carteira compra 10% dos candidatos pelo $V_T$ e é avaliada em $T+k$,
com as previsões fora da amostra do notebook 04. O **ROI da carteira** é $\sum\Delta V/\sum V_T$.

![Backtest](figures/05_model_interpretation/fig08_backtest_estrategias.png)

| ROI médio das temporadas | Modelo | Modelo prudente (P10) | *Momentum* | Juventude | Aleatória |
|---|---|---|---|---|---|
| $T+1$, todos os candidatos | **61%** | 33% | 28% | 27% | −3% |
| $T+1$, faixa de € 2–15 mi | **59%** | 41% | 35% | 30% | −2% |
| $T+2$, todos os candidatos | **153%** | 95% | 49% | 56% | −4% |
| $T+2$, faixa de € 2–15 mi | **113%** | 81% | 67% | 64% | −1% |

- **Superioridade em todas as combinações:** o modelo vence em todas as 14 combinações de
  temporada, universo e horizonte.
- **Taxa de valorização da carteira:** 53–57% em $T+1$ e 58–67% em $T+2$, contra cerca de 25% ao
  acaso.
- **Perdas graves (< −50%):** 3–4% em $T+1$.
- **Calibração da carteira:** o ROI previsto subestima o realizado (53% contra 61% em $T+1$; 103%
  contra 153% em $T+2$). O modelo é **conservador**.

## 8. Cenário de orçamento e risco

Clube médio: € 40 mi por janela, até 6 atletas de € 2–15 mi, comprados em ordem de retorno esperado.
A referência são 2.000 carteiras com ordem de compra aleatória. O risco do modelo é estimado por
*bootstrap* dos atletas selecionados.

![Orçamento e risco](figures/05_model_interpretation/fig09_orcamento_risco.png)

| Temporada ($T+1$) | Investimento | ROI do modelo | Percentil vs. aleatório | P(perda) do modelo |
|---|---|---|---|---|
| 2020/21 | € 19,0 mi | −3% | 64 | 57% |
| 2021/22 | € 16,5 mi | +185% | 99,7 | 0% |
| 2022/23 | € 20,5 mi | +215% | 99,9 | 0% |
| 2023/24 | € 17,0 mi | +77% | 96,3 | 16% |

**Carteiras aleatórias:**
- ROI médio próximo de zero;
- P(perda) de 57–67%;
- VaR a 5% de cerca de −38% e CVaR de cerca de −44%.

Em $T+2$, o modelo fica no percentil ≥ 99,9 em todas as temporadas.

**Implicação gerencial.** O modelo desloca a distribuição de retorno para a direita, mas uma carteira
de 6 atletas continua exposta a choques de mercado. Em 2020/21, logo após a revisão de valores da
pandemia, o ROI foi de −3%. A recomendação combina a previsão com o intervalo de 80% e com
diversificação por posição e faixa etária.

## 9. Mapa de oportunidades e lista de observação (2023/24)

![Mapa de oportunidades](figures/05_model_interpretation/fig10_mapa_oportunidades.png)

O mapa posiciona cada atleta pelo retorno esperado e pelo cenário pessimista (P10). Na lista das 15
maiores oportunidades de € 2–15 mi apontadas em 30/06/2024 (`tab09`), **53% se valorizaram** até
30/06/2025, e o **ROI da lista foi de 132%**.

| Destaques | $V_T$ → $V_{T+1}$ |
|---|---|
| Abdukodir Khusanov (Lens) | € 2,5 mi → € 35 mi |
| Jon Aramburu (Real Sociedad) | € 2,5 mi → € 15 mi |
| Senny Mayulu (PSG) | € 5 mi → € 15 mi |
| Conor Bradley (Liverpool) | € 15 mi → € 30 mi |
| Edan Diop (Monaco) | € 2 mi → € 0,6 mi |

O retorno vem de poucos acertos grandes, com perdas limitadas: o P10 fica em torno de −40%.

## 10. Conclusões para o artigo

1. A valorização futura é explicada por idade, preço, trajetória do valor e desempenho, nessa ordem
   de magnitude. O **gap de avaliação** é o determinante mais alinhado à lógica *buy-low*.
2. O desempenho importa mais no **auge da carreira**. Para veteranos, a idade domina, o que
   contraria a premissa da especificação.
3. A estratégia baseada no modelo **supera** heurísticas e o acaso em todas as temporadas, com ROI de
   valor de mercado de 59–61% em $T+1$ e 113–153% em $T+2$.
4. O risco de carteiras concentradas permanece e deve ser gerido com os intervalos de previsão.

**Ressalvas.** O SHAP descreve o modelo, não relações causais. O ROI é de **valor de mercado**, sem
custos de transação, salários nem a diferença entre valor estimado e preço pago. As simulações são
retrospectivas.
