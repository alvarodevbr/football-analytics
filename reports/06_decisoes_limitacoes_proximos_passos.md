# 06 — Decisões, hipóteses, limitações e próximos passos (versão 4)

Fecha a documentação técnica da versão 4. A v4 aplica a revisão externa `documents/ENHANCEMENTS.md`.
A versão 3 (pré-registrada em `5573d7e`) está em `archive/v3_2026-10-23/`, inclusive este relatório
na forma original.

## 1. Protocolo e independência do teste

- **v3:** protocolo `protocol_v3.json`, *commit* `5573d7e` (23/10/2026). O teste de $T+1$ (2024/25) foi
  revelado em 23/10/2026.
- **v4:**
  - protocolo `protocol_v4.json`, *commit* `8883da8` (17/11/2026), SHA-256 `8741f3bb4aac…`;
  - painel `9bfdbda6da`, 125 variáveis, *hash* do código `d0c1a7550400`, que agora cobre também
    `features.py` e `interpretation.py`;
  - o notebook 04 verificou o protocolo antes de calcular qualquer métrica do teste.
- **Independência:** todas as escolhas da v4 usaram só as dobras de desenvolvimento, mas os autores já
  conheciam o resultado da v3 em 2024/25. Os dois testes (2024/25 em $T+1$ e 2023/24 em $T+2$) são,
  portanto, **semi-independentes**, e os números da v3 ficam como referência (`tab05b_referencia_v3`).
- **Próxima verificação intocada:** a lista prospectiva 2024/25 → 30/06/2027
  (`tab12_lista_prospectiva_2425_T2`, emitida com os modelos de `8883da8`) e a temporada 2025/26, cujo
  alvo de $T+1$ só existe em 30/06/2027.

## 2. Escolhas pré-declaradas e o que elas escolheram

| Regra (protocolo v4) | Escolha nas dobras de desenvolvimento | Sustentou-se no teste? |
|---|---|---|
| Alvo: nominal ou excesso ao *drift* | nominal nos dois horizontes (o excesso ficou a 0,0002 de RMSE; o *drift* por regime, 0,02 pior) | — |
| Arquitetura: só troca o *ensemble* global com $p_{\text{Holm}} < 0{,}05$ (dois agrupamentos) | $T+1$: *ensemble* com interações restritas ($p_{\text{Holm}} < 0{,}001$); $T+2$: global (o *pooling* 3 × 3 da v3 caiu para $p_{\text{Holm}} = 0{,}08$) | sim: as duas são as melhores no teste |
| Camada de magnitude: menor erro por decil | $T+1$: forma multiquantil; $T+2$: fator heterocedástico | sim: menores erros no teste (14,8 e 17,9 p.p.) |
| Probabilidade de decisão: menor Brier | $T+1$: regressão; $T+2$: média dos classificadores | sim (Brier 0,132 e 0,131 contra 0,17 da taxa-base) |
| Carteira sob orçamento com penalidade de risco: λ no desenvolvimento | λ = 1 nos dois horizontes | não em $T+2$ (−33%, 3 atletas) |

## 3. Hipóteses

| Hipótese | Critério | Resultado no teste | Veredito |
|---|---|---|---|
| H1. Desempenho agrega ao Transfermarkt e à idade | retirar desempenho (T, T−1) e valor justo aumenta a RMSE, $p_{\text{Holm}} < 0{,}05$ | $T+1$: +0,011 ($p < 0{,}001$); $T+2$: +0,009 ($p = 0{,}07$) | **Sustentada em $T+1$** (v3: nos dois) |
| H2. 4 × 3 supera 3 × 3 e global | MCS com uma 4 × 3, sem global nem 3 × 3 | o MCS contém os *ensembles* globais e o Ridge | **Não sustentada** |
| H3. Subavaliados valorizam-se mais | quintis extremos do gap com ICs disjuntos | $T+1$: −10,0% [−14,3; 0,0] × −20,0% [−25,0; −16,7]; $T+2$: ICs tocam-se em −33,3% | **Sustentada em $T+1$** |
| H4. Carteira do modelo supera heurísticas e acaso | vence o acaso em todas as temporadas de $T+1$ e as heurísticas na média | 6/6 ($p$ = 0,016); ROI de 58% × 26% (10% mais jovens), −1% (jovens titulares), 9% (*momentum*), −19% (acaso) | **Sustentada** |
| H5. Desempenho pesa mais no auge | 24–28 menos 33+ no $\vert$SHAP$\vert$, IC > 0 | +9,6 p.p. [9,2; 10,0] e +9,5 p.p. [9,1; 9,9] | **Sustentada** |

## 4. Como cada item da revisão externa foi tratado

| Item de ENHANCEMENTS.md | Tratamento na v4 |
|---|---|
| 1.1 Defasagem de 183 dias | Já era pequena: mediana de 25 dias; Δt entre $T$ e $T+1$ com mediana de 365 dias (302–427). Na v4 entraram: a defasagem como variável; o alvo anualizado (robustez, sem efeito); o desempenho posterior à avaliação, implementado mas fora do vetor (Understat em manutenção). |
| 1.2 Contrato | Continua fora: não há fonte histórica sem vazamento. Tempo de casa e empréstimo seguem fora pela cobertura de 2024/25 (80%). |
| 1.3 Lesões e seleções | Lesões **incluídas** (5.136 atletas, truncadas na âncora). Convocações: só uma API interna do Transfermarkt traz as datas, e o acesso foi bloqueado; ficam como limitação. |
| 1.4 xGChain/xGBuildup e minutos por fonte | Já eram usados na v3 (a revisão estava errada); entrou a razão xGBuildup/xGChain. Clube em 30/jun e transferência no meio da temporada: implementados, dependem do Understat. |
| 1.5 Censura fora das Big Five | O histórico do Transfermarkt já era global (a revisão estava errada). A intenção de tratar passou a distinguir aposentadoria e falta de clube (valor zero) de avaliação antiga (último valor). |
| 2.1 Corte p90 | Mantido (escopo do CLAUDE.md); sensibilidade a tetos de € 5–25 mi (`tab16_sensibilidade_teto_preco`). |
| 2.2 Alvo sem *drift* | Testado e não escolhido pela regra (empate com o nominal). |
| 2.3 Prioris por papel tático e surpresa de cauda | Incluídas. |
| 2.4 Colinearidade e vazamento intraclube no valor justo | Validação cruzada agrupada por clube; valor justo fora do vetor; gap só de desempenho. A ideia de tirar a idade do gap não foi adotada: o gap já é resíduo de um modelo com idade. |
| 2.5 Trajetória T−2 → T | Incluída (titularização, chegada às Big Five, aceleração, anos desde o pico). |
| 3.1 Empate árvores × lineares | Diagnóstico feito: o empate **não** vem do empilhamento; sem os gaps, as árvores perdem mais que o Ridge. |
| 3.2 Segmentação dura | Arquitetura global com interações restritas, escolhida em $T+1$. Restrições monotônicas não adotadas (imporiam a H3). |
| 3.3 Cauda direita | Cinco camadas comparadas. O erro do 10º decil caiu de 187 para 134 p.p. ($T+1$, teste), mas o decil segue subestimado (+189% × +323%). |
| 3.4 Perda alinhada à decisão | Classificadores diretos e métricas de decisão pré-registradas. YetiRank não adotado (custo; o classificador cobre o objetivo). Coerência entre horizontes: correlação de 0,95–0,98 (`tab17`). |
| 4.1 Realismo da simulação | Custo de carregamento, preço de aquisição observado, universo comprável, ROI ponderado e simples. |
| 4.2 Mochila com risco conformal | Implementada (`interpretation.budget_milp`); a penalidade não se mostrou robusta. |
| 4.3 SHAP de coalizão e sem a nota | Coalizão etária na Figura 1; SHAP sem a nota em `tab01b`. |
| 4.4 Inferência bidirecional e heurística mais forte | Dois agrupamentos na regra de arquitetura; heurística "jovens titulares" pré-registrada na H4. |
| 5.x Editoriais | Todos aplicados no artigo (ver a lista no fim). |

## 5. Limitações

- **Dados:**
  - o valor do Transfermarkt é uma *proxy* do preço, sujeita a ancoragem;
  - a revisão de abril de 2020 afeta a 1ª dobra;
  - faltam o contrato histórico, as convocações e a forma por partida;
  - tempo de casa e empréstimo ficam fora pela cobertura de 2024/25;
  - o histórico de lesões pode ter sido editado depois da data-âncora.
- **Teste:** os dois testes são semi-independentes. O ganho da v4 sobre a v3 é pequeno ($R^2$ +0,008 e
  +0,011) e não tem teste formal próprio.
- **Magnitude:** o decil superior continua subestimado, e o erro individual permanece alto (erro absoluto
  mediano de 21 p.p. em $T+1$).
- **Simulação:**
  - ROI de valor de mercado, e não de caixa;
  - o preço de aquisição observado vem de vendas realizadas (seleção);
  - a penalidade de risco da mochila concentrou a carteira;
  - as maiores oportunidades do teste estão em elencos de elite. O universo comprável trata o problema,
    mas não o elimina.
- **Escolhas no desenvolvimento de $T+2$:** só a 3ª dobra tem histórico para escolher a camada de
  magnitude e a probabilidade de decisão (as anteriores são aquecimento ou purga).

## 6. Próximos passos

1. Conferir a lista prospectiva em julho de 2027 (*commit* `8883da8`).
2. Coletar as escalações do Understat quando o site voltar e testar o bloco de forma num protocolo v5,
   com 2025/26 como primeiro teste intocado.
3. Buscar fonte de contrato histórico sem vazamento e de convocações datadas.
4. Modelo específico para saltos extremos, com o classificador de saltos como ponto de partida.

## 7. Alterações editoriais aplicadas ao artigo (ENHANCEMENTS §5)

- **Taxas de acerto:** a bruta ($\Delta V\% > 0$) e a líquida ($\Delta V\% > 22{,}2\%$) aparecem sempre
  rotuladas, e a taxa-base tem uma casa decimal.
- **Arquitetura final e funil:**
  - a arquitetura final de cada horizonte vem explicada pela regra, com o $p$ de Holm;
  - a Tabela 2 traz uma linha de referência da v3;
  - a Tabela 1 tem a ponte 22.419 → 19.727 e as linhas de alvo observado.
- **Texto:**
  - "especificações" foi substituído por expressões autoexplicativas;
  - nota de rodapé com o repositório, os *commits* e o SHA-256 do protocolo;
  - seção de disponibilidade de dados e código.
- **Figuras:**
  - Figura 1: rótulo $\ln V_T$, convenção de cores do shap e coalizão etária;
  - Figura 2: título com $T+1$ e $T+2$, rótulos legíveis e nota com C1–C12;
  - Figura 3: ICs de Wilson e histograma das probabilidades.
- **Equações e resumo:**
  - equações (2), (4) e (6) logo após a frase que as introduz;
  - identificação sequencial de $a$, $b$ e $f_g$;
  - *Abstract* em inglês.
- **Introdução:** a distinção entre explicar o nível e prever a primeira diferença.
- **Resultados:**
  - IC de $T+2$ na H5;
  - "10% mais jovens" tratado como referência demográfica, com o alfa de *scouting*;
  - apêndices de sensibilidade ao teto, robustez e camadas de magnitude.
