# Índice de figuras

Legendas sugeridas (ABNT: título acima da figura, fonte abaixo).

**Figura 1 – Valores SHAP das 20 variáveis mais importantes para o log-retorno em T+1 (ensemble global)**  
`fig01_shap_beeswarm_T1.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Cada ponto é um atleta–temporada (amostra de 2.500). Cor: percentil do valor da variável (azul = alto). À direita de zero, a variável eleva a valorização prevista.

**Figura 2 – Valores SHAP das 20 variáveis mais importantes para o log-retorno em T+2 (ensemble global)**  
`fig02_shap_beeswarm_T2.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Cada ponto é um atleta–temporada (amostra de 2.500). Cor: percentil do valor da variável (azul = alto). À direita de zero, a variável eleva a valorização prevista.

**Figura 3 – Participação de cada grupo de variáveis na explicação SHAP da valorização, por categoria**  
`fig03_grupos_por_categoria.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Média de |SHAP| somada por grupo, como % do total de cada categoria (C1–C3: 16–23 anos; C4–C6: 24–28; C7–C9: 29–32; C10–C12: 33+; em cada trinca, DEF, MEI e ATA). Defasagem inclui o momentum do valor.

**Figura 4 – As cinco variáveis com maior contribuição média (|SHAP|) para a valorização em T+1, por categoria**  
`fig04_top5_por_categoria.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Linhas: faixas etárias (16–23, 24–28, 29–32, 33+). Colunas: defensores, meio-campistas e atacantes.

**Figura 5 – As dez métricas tático-técnicas com maior contribuição para a valorização em T+1, por posição**  
`fig05_desempenho_por_posicao.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: (+): valores altos da métrica elevam a valorização prevista; (−): reduzem.

**Figura 6 – Gráficos de dependência SHAP das principais variáveis para a valorização em T+1**  
`fig06_dependencia_shap.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Cada ponto é um atleta–temporada; extremos (1% de cada lado) omitidos.

**Figura 7 – Decomposição SHAP da previsão de valorização em T+1 para o atleta de maior valorização esperada em cada faixa etária (2023/24)**  
`fig07_waterfall_exemplos.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Previsões fora da amostra do teste final; contribuições do modelo final. Valores em log-retorno.

**Figura 8 – ROI realizado das carteiras de recrutamento por estratégia e temporada de decisão (backtest fora da amostra)**  
`fig08_backtest_estrategias.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Carteiras com 10% dos candidatos de cada temporada e universo. ROI = ΣΔV / ΣV_T. Haste na estratégia aleatória: intervalo de 5% a 95% de 2.000 sorteios.

**Figura 9 – Distribuição do ROI de carteiras com orçamento de € 40 milhões: modelo versus ordem de compra aleatória**  
`fig09_orcamento_risco.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Até 6 atletas com V_T entre € 2 e € 15 milhões. Histograma: 2.000 sorteios por temporada, temporadas agrupadas.

**Figura 10 – Mapa de oportunidades de recrutamento em 30/06/2024: retorno esperado versus cenário pessimista em T+1**  
`fig10_mapa_oportunidades.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Tamanho do ponto proporcional a V_T. Rótulos: oito maiores retornos esperados na faixa de € 2–15 milhões. Previsões fora da amostra.
