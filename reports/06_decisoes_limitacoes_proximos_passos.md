# 06 — Decisões Metodológicas, Hipóteses, Limitações e Próximos Passos

[← Índice](README.md)

Este relatório registra:
- as hipóteses da versão 3 e o veredito de cada uma;
- as decisões que desviam das especificações (`specs/`) e o motivo de cada uma;
- as limitações e as ameaças à validade a declarar no artigo;
- os próximos passos.

## 1. Hipóteses e vereditos

Os critérios de cada hipótese foram fixados no protocolo (`reports/protocol_v3.json`, *commit*
`5573d7e`) antes da revelação do teste de 2024/25. Os vereditos aplicam esses critérios sem ajuste.

| Hipótese | Critério pré-registrado (teste final) | Resultado | Veredito |
|---|---|---|---|
| **H1.** O desempenho tem poder preditivo incremental além do Transfermarkt (nível e trajetória do valor) e da idade | Retirar o bloco desempenho em $T$ e $T-1$ + valor justo aumenta a RMSE, com $p$ de Holm < 0,05 | $T+1$: +0,011 ($p_{Holm}$ < 0,001); $T+2$: +0,010 ($p_{Holm}$ = 0,015) | **Sustentada** |
| **H2.** A matriz 4 × 3 supera a 3 × 3 de Khalife et al. (2025) e o modelo global | O MCS (10%) contém uma arquitetura 4 × 3 e exclui o *ensemble* global e as 3 × 3 | O MCS contém o *ensemble* global e as arquiteturas 3 × 3 nos dois horizontes; a segmentação pura é pior que o global | **Não sustentada** |
| **H3.** Atletas subavaliados (gap < 0) valorizam-se mais | Mediana de $\Delta V\%$ do quintil mais subavaliado acima da do mais sobreavaliado, com ICs de 95% disjuntos | $T+1$: −5,9% [−12,5; 0,0] contra −20,0% [−25,0; −16,7]; $T+2$: −25,0% [−33,3; −16,7] contra −38,5% [−42,9; −33,3] | **Sustentada em $T+1$**; em $T+2$, não pelo critério estrito (os ICs se tocam em −33,3%) |
| **H4.** A carteira do modelo supera heurísticas e o acaso, líquida de custos e incluindo quem sai da base | No cenário principal, vence a carteira aleatória nas 6 temporadas de $T+1$ (teste do sinal, $p$ = 0,016) e supera, na média, *momentum* e juventude | 6 de 6 ($p$ = 0,016); ROI médio de 46%, contra 9% (*momentum*), 6,5% (juventude) e −19% (aleatória) | **Sustentada** |
| **H5.** O desempenho pesa mais no auge que entre os veteranos | Diferença de participação do desempenho no $\vert\phi\vert$ (24–28 menos 33+) com IC de 95% acima de zero | +10,4 p.p. [10,0; 10,8] em $T+1$; +10,5 p.p. [10,1; 10,9] em $T+2$ | **Sustentada** |

**Leitura conjunta.**
- **H1 e H5 formam o achado central.** O desempenho em campo carrega informação sobre a valorização
  futura que o próprio Transfermarkt ainda não incorporou, sobretudo no auge da carreira. O ganho
  preditivo é modesto: +0,03 de $R^2$ em $T+1$ sobre o modelo com ciclo de vida, preço e trajetória
  do valor (relatório 04, §7).
- **H2 contraria a inovação declarada nas especificações.** A heterogeneidade por idade e posição é
  real (H5), mas um modelo único que as recebe como variáveis a captura melhor que modelos separados,
  nas duas matrizes.
- **H3 confirma a premissa *buy-low* no horizonte de uma temporada.** No teste, a correlação parcial
  entre o gap e o log-retorno, controlada por idade e preço, é negativa nos dois horizontes: −0,20 em
  $T+1$ e −0,15 em $T+2$.
- **H4 depende de poucas decisões por horizonte.** Em $T+2$, vencer as 4 temporadas dá $p$ = 0,063,
  que não atinge 5% nem com 4 vitórias em 4. Nas temporadas de 2019/20 a 2023/24, as previsões vêm de
  dobras usadas na otimização. Só 2024/25 é inteiramente limpa: ROI de 87%, contra −16% ao acaso.
- **Ressalva sobre a H5.** O IC é condicional ao modelo final: mede a variação entre atletas e clubes,
  não a incerteza do ajuste. Nos modelos reajustados em cada dobra, a diferença fica entre 6,8 e 10,7
  p.p. (relatório 05, §6): o padrão não depende de uma temporada nem do modelo final.

**Análise secundária, não pré-registrada: o modelo como regra de compra** (relatório 05, §9.2). O uso
gerencial do modelo é decidir quem comprar. Por isso, a "valorização significativa" foi definida como a
que paga os custos do cenário principal ($\Delta V\%$ > 22,2%). O limiar decorre dos custos fixados no
protocolo, mas a análise foi definida depois da revelação do teste.
- **Discriminação.** No teste, o modelo separa quem paga a operação com AUC de 0,82 ($T+1$) e 0,83
  ($T+2$). Do seu decil superior, 67% e 62% pagaram a operação, contra uma taxa-base de 21–22%.
- **Idade.** Comprar os 10% mais jovens já atinge AUC de 0,80 e 0,84. O modelo supera essa heurística
  em $T+1$ (+0,020 [0,009; 0,031]) e empata em $T+2$ (−0,002 [−0,013; 0,009]). A pergunta "vai se
  valorizar?" é respondida em boa parte pela idade.
- **Contribuição própria do modelo.** Aparece no auge da carreira e no retorno. Contra os mais jovens,
  o ROI do cenário principal é de 46% contra 26% em $T+1$ e de 89% contra 55% em $T+2$, com vitória em
  todas as temporadas. A H4 pré-registrada não incluía essa referência, mais forte que a juventude
  sorteada; com ela, a conclusão se mantém.

## 2. Protocolo pré-registrado

O teste final de $T+1$ (2024/25, alvo em 30/06/2026) foi tratado como **pré-registrado**:

1. os notebooks 02 e 03 ocultam os alvos de $T+1$ de 2024/25 em todos os diagnósticos;
2. o notebook 04 decide tudo nas dobras de desenvolvimento (hiperparâmetros, $\kappa$ do *pooling*,
   arquitetura final) e grava `reports/protocol_v3.json`. O arquivo contém o hash do painel e do
   código, as variáveis, as dobras, os hiperparâmetros, as regras de decisão e os critérios das
   hipóteses;
3. o protocolo foi *commitado* em `5573d7e` (23/10/2026, 00h31), antes de qualquer métrica do teste;
4. a revelação (`FA_REVEAL_HOLDOUT=1`) só ocorre se o protocolo estiver *commitado* e idêntico ao que a
   execução reproduz. A execução de referência confirmou: "Protocolo verificado (commit 5573d7e)".

O teste final de $T+2$ (2023/24) é **semi-independente**: foi consultado durante o desenvolvimento das
versões 1 e 2, e o alvo de $T+2$ de 2024/25 (30/06/2027) ainda não existe.

**Decisões tomadas nas dobras de desenvolvimento, antes do protocolo:**
- a arquitetura final ($T+1$: *ensemble* global; $T+2$: *pooling* 3 × 3), pela regra pré-declarada;
- a exclusão da 1ª dobra do histórico das camadas de magnitude e de intervalo (relatório 04, §10).

## 3. Decisões que divergem das especificações

| # | Especificação | Decisão adotada | Motivo | Evidência |
|---|---|---|---|---|
| 1 | Métricas avançadas do FBref (passes e conduções progressivas, SCA, xG, duelos aéreos) | Sofascore e Understat como fontes complementares; *proxies* para passes progressivos, SCA e conduções | O FBref retirou os dados Opta em 20/01/2026, inclusive das temporadas históricas | Relatório 01, §2 |
| 2 | Limiar $p_{90}$ ≈ € 18 mi (Müller et al., 2017) | $p_{90}$ empírico por temporada (€ 27–32 mi); € 18 mi citado como referência histórica | € 18 mi corresponde ao percentil 81 nesta amostra | Relatório 02, §3 |
| 3 | $\Delta V\%$ como alvo primário | Treino no log-retorno, com avaliação também em $\Delta V\%$ e € | Assimetria de 19,8 e curtose de 662 em $\Delta V\%$ | Relatório 02, §7 |
| 4 | Faixas "24–28" e "28–32" | 16–23, 24–28, 29–32, 33+ (anos completos) | Os rótulos se sobrepõem em 28; essa é a leitura formal da Tabela 6.2 | Relatório 02, §1 |
| 5 | Tempo de contrato e *dummy* Bosman no vetor explicativo | Reconstruídos, mas usados só em análise de robustez (`lookahead`) | Toda reconstrução depende de eventos posteriores a $T$ e antecipa o desfecho | Relatório 03, §3; 04, §8 |
| 6 | Efeitos de temporada | Regime de mercado observado em $T$ | A temporada de teste nunca aparece no treino | Relatório 03, §4 |
| 7 | 12 modelos segmentados como arquitetura principal | Modelo global com a categoria como variável; segmentação 4 × 3 e 3 × 3 e *pooling* como alternativas testadas | H2 não sustentada: a segmentação pura é pior que o global | Relatório 04, §6 |
| 8 | Árvores superariam modelos lineares | *Ensemble* de árvores e Ridge | Árvores e lineares empatam no teste; o *ensemble* é o único sempre no conjunto de confiança | Relatório 04, §4 |
| 9 | $R^2$ de referência de 0,65–0,87 (Tabela 6.1) | $R^2$ de 0,33 ($T+1$, teste pré-registrado) e 0,37 ($T+2$) no log-retorno | Os valores de referência são de *nowcasting*; o valor justo deste estudo, que é *nowcasting*, atinge 0,79 | Relatório 04, §12 |
| 10 | Validação cruzada K-fold (Verificação dos Dados, §3.2) | Janela expandida com purga | O K-fold aleatório vaza informação temporal | Relatório 03, §8 |
| 11 | Amostra 2019/20–2023/24 | **2018/19–2024/25**, com 2017/18 só para defasagens | Teste final virgem para $T+1$ e 40% mais dados | Relatórios 01 e 02 |
| 12 | Teste final em 2023/24 | **2024/25 pré-registrado** em $T+1$; 2023/24 semi-independente em $T+2$ | O 2023/24 orientou as versões 1 e 2 | §2 |
| 13 | Espaços de busca da especificação (§6.6) | Taxa de aprendizado fixa (0,05), árvores por parada antecipada temporal, espaços centrados em regularização, 20 avaliações | Na v2, os ótimos caíam na fronteira dos espaços; custo do processador | Relatório 04, §2 |
| 14 | Vencedor por categoria | Algoritmo fixo por segmento, *pooling* parcial e regra de decisão com Holm | Na v2, o "vencedor" se espalhava entre os seis algoritmos, com $R^2 \approx 0$ | Relatório 04, §6 |
| 15 | Taxas por 90 cruas | **Encolhimento empírico-bayesiano** por temporada × posição | Ruído das taxas de quem joga pouco; validado pela estabilidade entre temporadas | Relatório 03, §5.1 |
| 16 | ClubElo (planejado na v3) | Classificação, saldo de gols, promoção/rebaixamento e participação europeia; Elo fora | API do ClubElo fora do ar (HTTP 502) | Relatório 01, §5 |
| 17 | Vaga europeia em $T+1$ (planejada na v3) | Participação europeia na própria temporada $T$ | Parte das vagas (*play-offs* de agosto) se decide depois da âncora | Relatório 01, §5 |
| 18 | Tempo de casa e empréstimo em $T$ (v2) | Fora do vetor pela regra de cobertura | Histórico de transferências não coletado para 19,7% de 2024/25 (bloqueio do Transfermarkt) | Relatório 03, §2 |
| 19 | Camada de magnitude com todas as dobras anteriores (v2) | Sem a 1ª dobra no histórico | A 1ª dobra treina com uma só temporada e tem erro de nível estrutural | Relatório 04, §10 |
| 20 | *Backtest* sem custos, condicionado à sobrevivência (v2) | Custos de transação, intenção de tratar e teste do sinal | Realismo e honestidade sobre perdas | Relatório 05, §9 |
| 21 | Premissa: o rendimento operacional governa o valor dos veteranos | Não confirmada | Nos atletas de 33+, a idade responde por 39–46% do $\vert\phi\vert$, e o desempenho pesa 10 p.p. menos que no auge (H5) | Relatório 05, §3 |
| 22 | Avaliação por erro de previsão ($R^2$, RMSE e MAE; Tabela 6.1) | Avaliação também como regra de compra: AUC, acerto do decil superior e probabilidade calibrada de pagar os custos ($\Delta V\%$ > 22,2%), com os 10% mais jovens como referência adicional | O objetivo gerencial é decidir quem comprar, e não acertar a magnitude de cada atleta; análise secundária, definida depois da revelação do teste | Relatório 05, §9.2 |

**Inconsistência nas especificações.** A numeração C1–C12 da Tabela 3.1 (*Metodologia*) é por coluna,
e a das Tabelas 5.2 e 6.2, por linha. Foi adotada a das Tabelas 5.2 e 6.2.

## 4. Limitações

### Dados
- **Valor de mercado como *proxy*.** O Transfermarkt segue o "princípio do juiz", com ancoragem e
  cascatas de informação (Herm et al., 2014). O *momentum* das reavaliações é, em parte, efeito do
  próprio processo de avaliação.
- **Choques da pandemia.** O alvo de $T+1$ de 2018/19 é a revisão extraordinária de abril de 2020 (82%
  dos atletas perderam valor), e o $V_T$ de 2019/20 parte desse valor deprimido.
- **Viés de sobrevivência.** $V_{T+2}$ falta em 22–26% dos atletas de 33+; os modelos treinam só com
  quem continua avaliado.
- **Contrato em $T$.** Não é observável retroativamente e foi reconstruído, com 45,8% de observação
  direta.
- **Coleta incompleta no Transfermarkt.** 652 históricos de transferência e 54 perfis de atletas novos
  em 2024/25 não foram coletados por bloqueio do site; tempo de casa e empréstimo saíram do vetor.
- **Força do clube sem Elo.** É representada por classificação, saldo de gols, valor do elenco e
  participação europeia. A classificação calculada não aplica punições administrativas de pontos.
- **Variáveis não coletadas:** convocações para a seleção, reputação internacional e lesões.
- **Minutagem mínima.** O filtro de 90 minutos seleciona jovens já integrados ao elenco principal.
- **Corte $p_{90}$ assimétrico.** Exclui 21% da Premier League e 4% da Ligue 1.

### Modelagem
- **Previsões encolhidas.** Os maiores saltos de valor continuam subestimados. No 10º decil de $T+1$, a
  esperança prevista é +96%, contra +194% observados. Esses saltos dependem de eventos não observáveis
  em $T$ (transferências, explosões de desempenho, convocações).
- **Pouca informação dentro das categorias.** O $R^2$ dentro de cada categoria vai de 0,01 a 0,16 em
  $T+1$ e é negativo nos meio-campistas e atacantes de 33+ em $T+2$. Boa parte da previsibilidade vem
  das diferenças entre faixas etárias.
- **Ganho incremental modesto do desempenho:** +0,03 de $R^2$ em $T+1$ e +0,02 em $T+2$.
- **Otimização com ganho desprezível e ótimos na fronteira** (XGBoost em $T+2$). A conclusão "algoritmo
  importa pouco" vale para os espaços testados.
- **Teste de $T+2$ semi-independente** (consultado nas versões 1 e 2).

### Interpretação e simulação
- **SHAP descreve o modelo, não relações causais.** Os efeitos de variáveis correlacionadas se dividem
  entre si; o encolhimento aumentou as correlações entre as métricas de desempenho.
- **ROI de valor de mercado, não de caixa.** Os custos de transação entram como cenário. Salários e a
  diferença entre valor estimado e preço efetivo não entram.
- **Carteiras retrospectivas**, salvo a lista prospectiva para 30/06/2027.
- **Regra de compra avaliada depois da revelação do teste.** Os rótulos de decisão, as métricas e a
  referência "Mais jovens" não constam do protocolo. Em $T+2$, a probabilidade de pagar a operação é
  otimista: acima de 60% previstos, 62–64% observados.

## 5. Ameaças à validade

| Tipo | Ameaça | Tratamento no estudo | Risco residual |
|---|---|---|---|
| Construto | O valor de mercado do Transfermarkt não é preço de transação; é uma avaliação de comunidade sob o "princípio do juiz" (Herm et al., 2014) | Alvo assumido como *proxy* consagrada (Müller et al., 2017; Peeters, 2018); o ROI do *backtest* é declarado como ROI de valor de mercado, com custos de transação como cenário | O modelo aprende também a inércia dos avaliadores (*momentum*, revisões), não só o "valor intrínseco" |
| Construto | As métricas do Sofascore e do Understat não são as da Opta | Fonte única por métrica em todas as temporadas; métricas sem cobertura histórica descartadas | Definições próprias de cada provedor (duelos, xG) |
| Interna | Vazamento de informação futura (*look-ahead*) | Âncora de 30/jun em toda *feature*; contrato reconstruído fora do vetor; vaga europeia de $T+1$ excluída; dobras com purga em $T+2$; asserts no notebook 03 | Rebaixamento lido pelo pertencimento em $T+1$ (fato decidido antes da âncora) |
| Interna | Otimismo da seleção de modelos (*garden of forking paths*) | Teste de $T+1$ pré-registrado; regras de decisão e critérios das hipóteses fixados antes da revelação; *ensemble* fixado a priori; robustez com hiperparâmetros-padrão | O teste de $T+2$ (2023/24) foi consultado nas versões 1 e 2; decisões de desenvolvimento (camada de magnitude, $\kappa$) usaram as mesmas dobras que avaliam |
| Interna | Viés de sobrevivência no alvo | Atrito reportado por categoria; *backtest* por intenção de tratar, com limites pessimista e otimista | Os modelos treinam só com quem continua avaliado; a depreciação dos veteranos tende a ser subestimada |
| Externa | Amostra restrita às Big Five, aos 90% mais baratos e a atletas com ≥ 90 min | Delimitação declarada; corte $p_{90}$ por temporada, com sensibilidade por liga | Não generaliza para ligas menores, superestrelas ou jovens sem minutagem na elite |
| Externa | Regime de mercado (pandemia; inflação) | Regime de mercado em $T$ como *feature*; modelo sem 2018/19 no treino como robustez | Choques futuros sem precedente na amostra |
| Conclusão estatística | Dependência entre atletas do mesmo clube e temporada | *Bootstrap* e erros-padrão por cluster clube × temporada | Clusters de uma mesma liga e temporada não são independentes entre si |
| Conclusão estatística | Comparações múltiplas | Correção de Holm por família; MCS para o conjunto de modelos | Análises exploratórias (SHAP por categoria) sem correção formal |
| Conclusão estatística | Seleção do $\kappa$ do *pooling* nas mesmas dobras do teste de decisão | Regra de decisão pré-declarada; confirmação no teste final | O $p$ de validação é otimista: a vantagem do *pooling* em $T+2$ não se repetiu no teste |
| Conclusão estatística | Poucas temporadas para o *backtest* | Teste do sinal entre temporadas e reamostragem dentro delas | Seis decisões em $T+1$ e quatro em $T+2$: poder limitado |
| Conclusão estatística | Avaliação de decisão definida depois da revelação do teste | Limiar derivado dos custos pré-registrados; nenhum modelo ou previsão alterado; a referência acrescentada (10% mais jovens) só torna a comparação mais difícil | A escolha dos rótulos e das métricas de decisão não foi pré-registrada |

## 6. Próximos passos

**Curto prazo**
1. **Escrever Resultados e Discussão** a partir dos relatórios 04 e 05 e do veredito das hipóteses.
2. **Uniformizar a numeração das categorias** e ajustar o texto das especificações sobre o $p_{90}$, o
   alvo, a segmentação e os $R^2$ de referência.
3. **Conferir a lista prospectiva** quando o Transfermarkt publicar os valores de junho de 2027.

**Dados (maior potencial de ganho)**

4. **Completar o Transfermarkt** quando o bloqueio terminar (652 históricos), numa versão seguinte com
   novo protocolo, já que o teste de 2024/25 foi revelado.
5. **Variáveis de visibilidade:** convocações, minutos em competições europeias e índices de reputação;
   *ratings* de potencial do EA FC (hipótese de Khalife et al., 2025).
6. **Histórico de lesões** e **base contratual histórica** (por exemplo, Capology), para medir o efeito
   Bosman sem *look-ahead*.

**Modelagem**

7. **Modelo em duas etapas para saltos de valor:** a probabilidade de uma revisão forte combinada com a
   magnitude condicional, com eventos de transferência tratados como cenários.
8. **Restrições monotônicas** (idade, gap) para a plausibilidade das explicações.
9. **Pré-registrar a avaliação de decisão** no protocolo da próxima versão: rótulo de pagar os custos,
   AUC, acerto do decil superior, calibração da probabilidade e a referência dos 10% mais jovens como
   critério da H4.

## 7. Referências metodológicas incluídas na versão 3

BERGMEIR, C.; HYNDMAN, R. J.; KOO, B. A note on the validity of cross-validation for evaluating autoregressive time series prediction. **Computational Statistics & Data Analysis**, v. 120, p. 70–83, 2018.

CAMERON, A. C.; GELBACH, J. B.; MILLER, D. L. Bootstrap-based improvements for inference with clustered errors. **The Review of Economics and Statistics**, v. 90, n. 3, p. 414–427, 2008.

CAMPBELL, J. Y.; THOMPSON, S. B. Predicting excess stock returns out of sample: can anything beat the historical average? **The Review of Financial Studies**, v. 21, n. 4, p. 1509–1531, 2008.

CERQUEIRA, V.; TORGO, L.; MOZETIČ, I. Evaluating time series forecasting models: an empirical study on performance estimation methods. **Machine Learning**, v. 109, p. 1997–2028, 2020.

CLAYTON, D.; KALDOR, J. Empirical Bayes estimates of age-standardized relative risks for use in disease mapping. **Biometrics**, v. 43, n. 3, p. 671–681, 1987.

COATES, D.; PARSHAKOV, P. The wisdom of crowds and transfer market values. **European Journal of Operational Research**, v. 301, n. 2, p. 523–534, 2022.

DIEBOLD, F. X.; MARIANO, R. S. Comparing predictive accuracy. **Journal of Business & Economic Statistics**, v. 13, n. 3, p. 253–263, 1995.

DUAN, N. Smearing estimate: a nonparametric retransformation method. **Journal of the American Statistical Association**, v. 78, n. 383, p. 605–610, 1983.

EFRON, B.; MORRIS, C. Data analysis using Stein's estimator and its generalizations. **Journal of the American Statistical Association**, v. 70, n. 350, p. 311–319, 1975.

FRANCESCHI, M. et al. Determinants of football players' valuation: a systematic review. **Journal of Economic Surveys**, 2024. (Conferir volume e páginas na versão publicada.)

GNEITING, T. Making and evaluating point forecasts. **Journal of the American Statistical Association**, v. 106, n. 494, p. 746–762, 2011.

HANSEN, P. R.; LUNDE, A.; NASON, J. M. The model confidence set. **Econometrica**, v. 79, n. 2, p. 453–497, 2011.

HARVEY, D.; LEYBOURNE, S.; NEWBOLD, P. Testing the equality of prediction mean squared errors. **International Journal of Forecasting**, v. 13, n. 2, p. 281–291, 1997.

HERM, S.; CALLSEN-BRACKER, H.-M.; KREIS, H. When the crowd evaluates soccer players' market values: accuracy and evaluation attributes of an online community. **Sport Management Review**, v. 17, n. 4, p. 484–492, 2014.

HOLM, S. A simple sequentially rejective multiple test procedure. **Scandinavian Journal of Statistics**, v. 6, n. 2, p. 65–70, 1979.

KAPOOR, S.; NARAYANAN, A. Leakage and the reproducibility crisis in machine-learning-based science. **Patterns**, v. 4, n. 9, 100804, 2023.

LEI, J. et al. Distribution-free predictive inference for regression. **Journal of the American Statistical Association**, v. 113, n. 523, p. 1094–1111, 2018.

MARSHALL, R. J. Mapping disease and mortality rates using empirical Bayes estimators. **Journal of the Royal Statistical Society: Series C (Applied Statistics)**, v. 40, n. 2, p. 283–294, 1991.

McHALE, I. G.; HOLMES, B. Estimating transfer fees of professional footballers using advanced performance metrics and machine learning. **European Journal of Operational Research**, v. 306, n. 1, p. 389–399, 2023.

PEETERS, T. Testing the Wisdom of Crowds in the field: Transfermarkt valuations and international soccer results. **International Journal of Forecasting**, v. 34, n. 1, p. 17–29, 2018.

ROMANO, Y.; PATTERSON, E.; CANDÈS, E. Conformalized quantile regression. *In*: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS, 32., 2019, Vancouver. **Proceedings** [...]. Red Hook: Curran Associates, 2019. p. 3543–3553.

SMITH, J.; WALLIS, K. F. A simple explanation of the forecast combination puzzle. **Oxford Bulletin of Economics and Statistics**, v. 71, n. 3, p. 331–355, 2009.
