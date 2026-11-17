# Índice de figuras

Legendas sugeridas (ABNT: título acima da figura, fonte abaixo).

**Figura 1 – Convergência da otimização bayesiana dos hiperparâmetros dos modelos globais**  
`fig01_convergencia_otimizacao.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: À esquerda da linha vertical, pontos iniciais aleatórios; à direita, pontos escolhidos pelo processo gaussiano. RMSE média das três últimas dobras de validação.

**Figura 2 – Desempenho dos modelos globais no teste final: erro e capacidade de ordenação**  
`fig02_comparacao_modelos_teste.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: T+1: 2024/25 (pré-registrado); T+2: 2023/24. Hastes: IC de 95% por bootstrap de clubes × temporada.

**Figura 3 – Aumento do erro ao retirar cada bloco de variáveis do ensemble (teste final)**  
`fig08_ablacao_blocos.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Azul: diferença significativa após a correção de Holm (p < 0,05, erro-padrão por clube × temporada).

**Figura 4 – R² do log-retorno por categoria (matriz 4 × 3) e arquitetura**  
`fig03_r2_por_categoria.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Todas as dobras de teste agrupadas. Global: ensemble com faixa etária e posição como variáveis; Pooling: segmento encolhido para o global.

**Figura 5 – RMSE do log-retorno por temporada de teste nos modelos globais**  
`fig04_estabilidade_temporal.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Dobras de desenvolvimento e teste final.

**Figura 6 – Taxa de valorização observada por decil de valorização prevista (modelo final)**  
`fig05_calibracao_decis.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Decis calculados dentro de cada temporada de teste. A linha horizontal é a taxa média.

**Figura 7 – Variação percentual média observada e prevista por decil de valorização prevista**  
`fig06_magnitude_decis.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Dobras de teste a partir da 3ª (a camada usa só dobras anteriores, sem a de aquecimento).

**Figura 8 – Intervalo de previsão de 80% (P10–P90) e distribuição observada da variação percentual por decil**  
`fig07_intervalos_previsao.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Quantis do CatBoost conformalizados com as dobras anteriores; dobras de teste a partir da 3ª.
