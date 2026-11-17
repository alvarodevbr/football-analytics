# Índice de figuras

Legendas sugeridas (ABNT: título acima da figura, fonte abaixo).

**Figura 1 – Origem da informação sobre o contrato vigente em T, por temporada de referência**  
`fig01_origem_contrato.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Observado: término conhecido sem estimativa. Estimado: mediana do estrato tempo de casa × faixa etária.

**Figura 2 – Tempo restante de contrato em T e proporção de atletas no último ano de contrato (status Bosman)**  
`fig02_contrato_bosman.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Em (b), inclui casos estimados.

**Figura 3 – Correlação de Spearman entre as métricas de desempenho em T, ordenadas por agrupamento hierárquico**  
`fig03_correlacao_features_desempenho.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Distância 1 − |ρ|, ligação média. Blocos escuros indicam multicolinearidade.

**Figura 4 – Estabilidade de um ano para o outro das métricas de desempenho, taxa crua versus encolhida**  
`fig07_estabilidade_encolhimento.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Jogadores de linha com ≥ 90 minutos nas duas temporadas consecutivas. Encolhimento empírico-bayesiano por temporada × posição; nenhuma variável-alvo é usada.

**Figura 5 – Variação mediana do valor entre T e T+1 segundo o quintil da variação passada, por faixa etária**  
`fig04_momentum.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Variação mediana = exp(mediana do log-retorno) − 1. Quintis calculados sobre a amostra inteira.

**Figura 6 – Variação mediana do valor entre T e T+1 segundo o quintil do gap de avaliação em T, por faixa etária**  
`fig04b_gap_avaliacao.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Gap = ln(V_T) − valor justo estimado pelo desempenho em T (corte transversal, validação cruzada).

**Figura 7 – Esquema da validação por janela expandida nos horizontes T+1 e T+2**  
`fig05_janela_expandida.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Números: observações com alvo observado em cada temporada de referência.

**Figura 8 – Features com mais de 0,5% de valores ausentes no painel de modelagem**  
`fig06_ausentes_features.png`  
Fonte: Elaboração própria com dados de FBref, Understat, Sofascore e Transfermarkt (2026). Nota: Ausentes são imputados dentro de cada dobra no notebook 04 (mediana por posição, só com o treino).
