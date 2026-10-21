# 06 — Decisões Metodológicas, Limitações e Próximos Passos

[← Índice](README.md)

Este relatório registra as decisões que desviam das especificações (`specs/`) e o motivo de cada
uma, as limitações a declarar no artigo e os próximos passos recomendados.

## 1. Decisões que divergem das especificações

| # | Especificação | Decisão adotada | Motivo | Evidência |
|---|---|---|---|---|
| 1 | Métricas avançadas do FBref (passes e conduções progressivas, SCA, xG, duelos aéreos) | Sofascore e Understat como fontes complementares; *proxies* para passes progressivos, SCA e conduções | O FBref retirou os dados Opta em 20/01/2026, inclusive das temporadas históricas | Relatório 01, §2 |
| 2 | Limiar $p_{90}$ ≈ € 18 mi (Müller et al., 2017) | $p_{90}$ empírico por temporada (€ 27–32 mi); € 18 mi citado como referência histórica | € 18 mi corresponde ao percentil 81 nesta amostra | Relatório 02, §3 |
| 3 | $\Delta V\%$ como alvo primário | Treino no log-retorno, com avaliação também em $\Delta V\%$ e € | Assimetria de 19,5 e curtose de 634 em $\Delta V\%$ | Relatório 02, §7 |
| 4 | Faixas "24–28" e "28–32" | 16–23, 24–28, 29–32, 33+ (anos completos) | Os rótulos se sobrepõem em 28; essa é a leitura formal da Tabela 6.2 | Relatório 02, §1 |
| 5 | Tempo de contrato e dummy Bosman no vetor explicativo | Reconstruídos, mas usados só em análise de robustez (`lookahead`) | Toda reconstrução depende de eventos posteriores a $T$ e antecipa o desfecho | Relatório 03, §3; 04, §8 |
| 6 | Efeitos de temporada | Regime de mercado observado em $T$ | A temporada de teste nunca aparece no treino | Relatório 03, §4 |
| 7 | 12 modelos segmentados como arquitetura principal | Modelo global com categoria como variável; os segmentados ficam como comparação | O global é superior, com significância em 6 de 10 comparações | Relatório 04, §3 |
| 8 | Árvores superariam modelos lineares | *Ensemble* de árvores e Ridge | Árvores e lineares empatam; o *ensemble* é o único ganho significativo | Relatório 04, §3 |
| 9 | Espaços de busca: CatBoost profundidade 10 e 1.200 iterações; RF 800 árvores | Profundidade ≤ 8, ≤ 800 iterações ou árvores, RF ≤ 500; 10–15 avaliações por otimização | Custo computacional do processador disponível | Relatório 04, §1 |
| 10 | $R^2$ de referência de 0,65–0,87 (Tabela 6.1) | $R^2$ de 0,28 ($T+1$) e 0,37 ($T+2$) no log-retorno | Os valores de referência são de *nowcasting*; o *nowcasting* deste estudo atinge 0,75 | Relatório 04, §9 |
| 11 | Validação cruzada K-fold (Verificação dos Dados, §3.2) | Janela expandida com purga, conforme a Metodologia e o Design do Modelo | O K-fold aleatório vaza informação temporal | Relatório 03, §8 |
| 12 | Premissa: o rendimento operacional governa o valor dos veteranos | Não confirmada | SHAP: nos atletas de 33+, a idade explica cerca de 40% e o desempenho pesa menos que no auge | Relatório 05, §3 |

**Inconsistência nas especificações.** A numeração C1–C12 da Tabela 3.1 (*Metodologia*) é por coluna,
e a das Tabelas 5.2 e 6.2, por linha. Foi adotada a das Tabelas 5.2 e 6.2. Convém uniformizar no
artigo.

## 2. Limitações

### Dados
- **Valor de mercado como *proxy*.** O Transfermarkt segue o "princípio do juiz", com ancoragem e
  cascatas de informação (Herm et al., 2014). O *momentum* das reavaliações é, em parte, efeito do
  próprio processo de avaliação.
- **Revisão da pandemia.** 99% dos $V_T$ de 2019/20 datam de 08/04/2020, a revisão extraordinária da
  pandemia. Isso gera um efeito de regime nessa temporada.
- **Viés de sobrevivência.** $V_{T+2}$ falta em 24% dos atletas de 33+, e a depreciação dos
  veteranos tende a ser subestimada.
- **Contrato em $T$.** Ele não é observável retroativamente e foi reconstruído com 46,5% de
  observação direta.
- **Variáveis não coletadas:** convocações para a seleção, reputação internacional e lesões.
- **Minutagem mínima.** O filtro de 90 minutos seleciona jovens já integrados ao elenco principal.
- **Corte $p_{90}$ assimétrico.** Exclui 21% da Premier League e 4% da Ligue 1.
- **Coleta pendente.** 44 atletas pareados manualmente ainda não têm dados do Transfermarkt, por
  causa do bloqueio de acesso. São cerca de 70 observações sem $V_T$.

### Modelagem
- **Previsões encolhidas.** Os maiores saltos de valor continuam subestimados (inclinação de
  calibração > 1). Eles dependem de eventos não observáveis em $T$, como transferências, explosões
  de desempenho ou convocações.
- **Orçamento de otimização reduzido.** Em parte dos algoritmos, o melhor ponto foi encontrado ainda
  na fase aleatória.
- **Pouca informação dentro das categorias.** O $R^2$ intracategoria é baixo, e boa parte da
  previsibilidade vem das diferenças entre faixas etárias.

### Interpretação e simulação
- **SHAP descreve o modelo, não relações causais.** Os efeitos de variáveis correlacionadas se
  dividem entre si.
- **ROI de valor de mercado, não de caixa.** A simulação não inclui custos de transação, salários,
  comissões nem a diferença entre valor estimado e preço efetivo. As carteiras são retrospectivas.

## 3. Próximos passos recomendados

**Curto prazo**
1. **Recoletar os 44 jogadores pendentes.** Quando o Transfermarkt liberar o acesso, teste primeiro
   uma única requisição e depois reexecute os notebooks 01 → 05, nesta ordem. Os caches tornam a
   reexecução incremental.
2. **Escrever as seções de Resultados e Discussão** a partir destes relatórios e das leituras
   interpretativas dos notebooks.
3. **Uniformizar a numeração das categorias** e ajustar o texto da especificação sobre o $p_{90}$, o
   alvo e os $R^2$ de referência.

**Robustez para o capítulo de Resultados**
4. Corte $p_{90}$ por liga e temporada.
5. Modelos sem a temporada 2019/20.
6. Maior orçamento de otimização, numa máquina mais rápida.
7. Carteiras com restrição de diversificação (posição e faixa etária) no cenário de orçamento.

**Trabalho futuro (maior potencial de ganho na magnitude)**
8. **Variáveis de visibilidade:** convocações para a seleção, minutos em competições europeias e
   índices de reputação.
9. **Histórico de lesões.**
10. **Base contratual histórica** (por exemplo, Capology), para medir o efeito Bosman sem
    *look-ahead*.
11. **Modelo em duas etapas para saltos de valor:** a probabilidade de uma revisão forte combinada
    com a magnitude condicional, e eventos de transferência tratados como cenários.
