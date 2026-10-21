3. Problema e Objetivos

3.1. Contexto de Negócios e o Modelo Player Trading

A sustentabilidade financeira no futebol profissional europeu tornou-se um tema central no âmbito da economia do esporte e das finanças corporativas aplicadas. A gestão contemporânea de entidades desportivas enfrenta pressões estruturais decorrentes de inflação persistente no mercado de transferências, espirais de gastos operacionais (overspending) e busca desmedida por sucessos desportivos de curto prazo, fatores que frequentemente comprometem a liquidez e a solvência de longo prazo dos clubes (Martín-Magdalena et al., 2024; Hill et al., 2025). Diante desse cenário de escrutínio regulatório rigoroso — intensificado pelos regulamentos de sustentabilidade financeira da UEFA —, a geração de superávit operacional e a gestão eficiente do balanço patrimonial converteram-se em imperativos estratégicos indispensáveis.

Para responder a essas restrições orçamentárias, clubes de médio porte e organizações focadas no desenvolvimento de talentos — tais como o FC Midtjylland e o TSG Hoffenheim — reestruturaram seus modelos de negócios em torno do player trading pautado em inteligência de dados (analytics) (Müller et al., 2017). Essa abordagem operacional fundamenta-se na estratégia buy-low sell-high: a identificação e aquisição sistemática de ativos subavaliados no momento T, seguida pelo seu desenvolvimento técnico-tático e posterior alienação com expressivo ágio de transferência em janelas futuras (T+1 ou T+2).

A fundamentação teórica dessa estratégia encontra amparo direto nas tipologias de valoração de Damodaran adaptadas ao contexto esportivo por Hill et al. (2025). Modelos clássicos de valoração intrínseca baseados no Fluxo de Caixa Descontado (DCF), amplamente empregados na avaliação de empresas corporativas, apresentam limitações severas quando aplicados individualmente a atletas de futebol. A impossibilidade metodológica de isolar os fluxos de caixa operacionais gerados por um único jogador no ecossistema coletivo do jogo, a natureza intrinsecamente não geradora de caixa dos direitos contratuais do atleta e o fato de o valor residual (terminal value) anular-se ao término do contrato contratual — como decorrência direta do advento do Regulamento Bosman (Antonioni & Cubbin, 2000; Simmons, 1997) — inviabilizam a valoração intrínseca estrita (Hill et al., 2025).

Por conseguinte, clubes desenvolvedores operam sob a lógica das tipologias Relativa e Probabilística de Damodaran (Hill et al., 2025). Em um ambiente de maximização de utilidade esportiva sob severas restrições orçamentárias, a eficácia do recrutamento deixa de ser mensurada pelo montante absoluto dispendido e passa a ser avaliada pelo retorno de desempenho esportivo alcançado relativamente ao custo de aquisição do ativo (Berri et al., 2023; Hill et al., 2025).

O Quadro 1 sintetiza as dimensões fundamentais do modelo de player trading sustentável sob a perspectiva analítico-financeira.

Quadro 1 — Dimensões do Modelo de Negócios de Player Trading Sustentável.

Dimensão do Modelo de Negócios	Abordagem de Player Trading Sustentável
Objetivo Primário	Maximizar a utilidade esportiva e financeira mediante a captura de ágio monetário (capital gain) entre as janelas de aquisição (T) e revenda (T+1 ou T+2).
Métricas de Sucesso	Avaliação do desempenho esportivo entregue per capita em relação ao custo total de aquisição e manutenção salarial do atleta (retorno relativo ao custo).
Gestão de Risco de Mercado	Mitigação da assimetria informacional por meio da precificação estatística ex-ante de ativos subavaliados na compra e projeção do risco de volatilidade de desempenho.
Estratégia de Recrutamento	Mapeamento quantitativo sistemático (data-driven scouting) direcionado a atletas com assimetria favorável entre preço de mercado em T e potencial de valorização futura.

Fonte: Elaborado com base em Müller et al. (2017) e Hill et al. (2025).

3.2. Delimitação da População e Recorte Estratégico

A definição da população e os critérios de amostragem desta pesquisa respondem a exigências técnicas de estabilidade estatística e rigor teórico, articulando os dados empíricos de desempenho de campo da plataforma FBref com os pressupostos estabelecidos pela literatura de valoração de atletas.

* Justificativa do Recorte dos 90% (Exclusão do Top 10% / Superestrelas): A amostra do estudo é restrita aos 90% jogadores de linha mais baratos do mercado no momento T. Esta delimitação ampara-se nas descobertas empíricas de Müller et al. (2017), que demonstraram que modelos estatísticos quantitativos superam o desempenho preditivo das estimativas coletivas do mercado (crowdsourcing) precisamente no percentil abaixo de 90% — o qual correspondeu, na base empírica observada pelos autores, ao limite aproximado de €18 milhões na taxa de transferência. O segmento correspondente ao Top 10% mais caro (superestrelas) é fortemente inflacionado por determinantes extrapólio, como apelo comercial, valor de marca (brand equity), poder de atração de público e intangíveis de visibilidade internacional (Müller et al., 2017; Khalife et al., 2025). Tais ativos são inatingíveis ao orçamento e totalmente desalinhados da dinâmica operacional dos clubes desenvolvedores, cujo modelo de negócio depende da margem de valorização obtida sobre atletas de menor custo inicial.
* Restrição às Ligas Big Five Europeias: A amostragem abrange atletas atuarilados nas cinco principais ligas do continente europeu (Inglaterra - Premier League, Espanha - La Liga, Alemanha - Bundesliga, Itália - Serie A e França - Ligue 1). Este recorte garante elevado nível de homogeneidade técnica, padronização na coleta de métricas de alto rendimento e comparabilidade direta entre os ambientes competitivos.
* Filtro de Minutagem Mínima no FBref (\ge 90\text{ min}): O universo amostral é filtrado para incluir estritamente jogadores de linha que tenham acumulado no mínimo 90 minutos jogados na temporada de referência T na base do FBref. Este critério assegura a estabilidade variacional dos atributos técnicos e físicos observados (métricas normalizadas por 90 minutos, per-90), eliminando ruídos estatísticos severos e distorções causadas por amostragens nulas ou insuficientes de atletas sem tempo produtivo em campo.

3.3. Definição do Problema de Pesquisa: Da Assimetria ao Forecasting Temporal

A literatura especializada em economia do esporte postula uma distinção conceitual indispensável entre os termos Valor e Preço (Hill et al., 2025). O Valor de mercado de um atleta reflete seu patrimônio produtivo intrínseco e relativo, sendo estritamente derivado de seu rendimento tático-técnico em campo, atributos físicos e variáveis demográficas (como idade e experiência acumulada). Em contrapartida, o Preço pago em uma transação efetiva de transferência engloba variáveis circunstanciais, regulatórias e negociais, tais como a duração remanescente do contrato de trabalho, a urgência financeira da equipe vendedora, a liquidez de caixa do comprador e o poder de barganha relativo entre as partes (Hill et al., 2025). Para garantir a integridade da valoração intrínseca no momento T, as variáveis preditivas utilizadas pelo modelo para estimar o valor de mercado devem excluir determinantes estritamente contratuais de preço (como tempo restante de contrato), evitando a contaminação e a fusão indevida entre os conceitos de valor e preço.

Por outro lado, as abordagens tradicionais de valoração pautadas em estimativas de multidão (crowdsourcing), largamente disseminadas pela plataforma Transfermarkt, exibem fragilidades metodológicas acentuadas. Sob a ótica de Surowiecki (2004), a eficácia da "Sabedoria das Multidões" (Wisdom of Crowds) exige o cumprimento estrito de cinco pilares: diversidade de opinião, independência, descentralização, agregação e confiança. O ecossistema do Transfermarkt viola frontalmente o critério de independência, uma vez que os avaliadores da comunidade possuem acesso prévio às estimativas históricas e às opiniões de outros membros. Essa visibilidade encadeia vieses cognitivos sistemáticos, destacando-se a heurística de ancoragem (anchoring heuristic) (Tversky & Kahneman, 1974) e as cascatas de informação (informational cascades) (Bikhchandani et al., 1992; Hill et al., 2025; Müller et al., 2017), em que avaliadores subsequentes ajustam suas estimativas a partir do valor-âncora pré-existente.

Além dessas fragilidades estruturais, a literatura predominantemente foca na valoração estática ou instantânea (nowcasting). Para a execução prática do modelo de player trading, todavia, o nowcasting revela-se insuficiente. A sustentabilidade de um clube desenvolvedor exige a transição metodológica para a projeção preditiva temporal (forecasting) nos horizontes T+1 e T+2. Torna-se imperativo identificar ex-ante quais ativos subavaliados no presente demonstrarão variação positiva (absoluta monetária em Euros e relativa percentual) em janelas de transferências futuras (Hill et al., 2025; Khalife et al., 2025).

Para assegurar o rigor técnico e a validade causal do forecasting, o processo de modelagem de séries temporais deve instituir barreiras estritas contra o vazamento de dados (data leakage) (Kaufman et al., 2011; Müller et al., 2017). Dessa forma, os algoritmos treinados com atributos do momento T (métricas FBref per-90, dados demográficos e valor de referência) devem projetar as valorações dos horizontes T+1 e T+2 estritamente sem a incorporação de informações de desempenho acumuladas em temporadas futuras, preservando a integridade temporal do processo de amostragem.

3.4. Pergunta Central de Pesquisa

Alinhada às lacunas conceituais e metodológicas identificadas, formula-se a seguinte Pergunta Central de Pesquisa:

"Como prever a variação do valor de mercado futuro (absoluta e percentual) nos horizontes T+1 e T+2 para os 90% jogadores de linha mais baratos das Big Five europeias (com ao menos 90 min em T), a partir de atributos de campo do FBref, métricas contratuais e demográficas, visando fundamentar estratégias de recrutamento e revenda rentável?"

3.5. Objetivos da Pesquisa

3.5.1. Objetivo Geral

Desenvolver e validar modelos estatísticos e de Machine Learning capazes de prever a variação relativa percentual (%) e absoluta monetária (\text{EUR } €) do valor de mercado nos horizontes temporais T+1 e T+2 para os 90% jogadores de linha mais baratos atuantes nas ligas Big Five europeias (com minutagem em campo \ge 90 minutos em T), provendo uma estrutura quantitativa robusta para subsidiar decisões estratégicas de recrutamento e player trading rentável.

3.5.2. Objetivos Específicos e Estrutura de Segmentação (12 Categorias)

A relevância preditiva dos atributos de desempenho varia substancialmente conforme a função tática e o estágio da carreira em que o atleta se encontra (Khalife et al., 2025; Hill et al., 2025). Enquanto a valoração de atletas jovens é amplamente dominada pelo potencial futuro projetado e pela reputação internacional, a valoração de atletas em estágios intermediários e avançados da carreira desloca-se decisivamente para métricas técnicas consolidadas e específicas de cada posição — como eficiência de finalização para atacantes, métricas de passe e criação de chances para meio-campistas, e capacidade de desarme, ações defensivas e velocidade/ritmo (pace) para defensores (Khalife et al., 2025).

A fim de capturar essa heterogeneidade sem homogeneizar indevidamente a amostra, este estudo adota uma modelagem segmentada dinâmica estruturada em 12 categorias combinadas. Esta matriz representa uma extensão adaptativa do arcabouço metodológico de Khalife et al. (2025) — que originalmente propuseram uma matriz 3 \times 3 composta por 9 modelos (3 faixas etárias: 16–23, 24–29, 30+ \times 3 posições: Atacante, Meio-Campista, Defensor). A matriz 4 \times 3 proposta neste estudo (Tabela 1) detalha os ciclos de transição fisiológica e de desempenho em quatro estágios de carreira (Sub-23 / Early, Mid / Prime, Late / Experiente e Veteran), otimizando a precisão preditiva dos algoritmos.

Tabela 1 — Matriz de Segmentação em 12 Categorias Combinadas do Estudo (Career Stage \times Posição).

Faixa Etária (Career Stage)	Posição: Defensor	Posição: Meio-Campista	Posição: Atacante
16–23 anos (Early / Sub-23)	Categoria 1	Categoria 2	Categoria 3
24–28 anos (Mid / Prime)	Categoria 4	Categoria 5	Categoria 6
28–32 anos (Late / Experiente)	Categoria 7	Categoria 8	Categoria 9
32+ anos (Veteran)	Categoria 10	Categoria 11	Categoria 12

Fonte: Elaborado a partir da extensão da estrutura de segmentação de Khalife et al. (2025).

Frente a este arcabouço estruturado, definem-se os seguintes Objetivos Específicos:

1. Tratar, filtrar e integrar a base de dados de atributos estatísticos de desempenho em campo (per-90 do FBref), variáveis demográficas e indicadores de contexto dos atletas elegíveis pertencentes ao segmento dos 90% mais baratos da população.
2. Identificar e ranquear a importância relativa das variáveis preditivas (feature importance) para cada uma das 12 categorias combinadas de idade e posição, quantificando a transição dinâmica dos pesos das variáveis ao longo da carreira do atleta — avaliando a preponderância da reputação e do potencial projetado no estágio Early/Sub-23 e a migração para métricas técnicas de campo (shooting/dribbling para atacantes, passing/vision para meio-campistas, e defending/tackling/pace para defensores) nas etapas maduras (Khalife et al., 2025).
3. Treinar, calibrar e comparar algoritmos preditivos estatísticos e de Machine Learning (incluindo modelos de Gradient Boosting) para mensurar a variação do valor de mercado nos horizontes temporais T+1 e T+2 (isolando a variação percentual relativa e a variação absoluta em Euros), assegurando a prevenção do vazamento de dados (data leakage) mediante particionamento temporal estrito.
4. Validar empiricamente a capacidade dos modelos em identificar ativos subavaliados com alto potencial de valorização futura (buy-low sell-high), avaliando o desempenho preditivo por meio de validação cruzada (K-fold cross-validation) e comparando as métricas de erro estatístico — R^2 Ajustado, Erro Quadrático Médio (RMSE) e Erro Absoluto Médio (MAE) — entre as 12 categorias segmentadas.

3.6. Justificativa e Contribuição do Estudo

A relevância deste trabalho fundamenta-se em contribuições originais nos domínios acadêmico e prático:

* Contribuição Teórico-Acadêmica: O estudo preenche uma lacuna relevante na literatura de economia do esporte ao integrar as tipologias de valoração de Damodaran ao mercado de transferências (Hill et al., 2025). A pesquisa avança o estado da arte ao superar a valoração estática instantânea (nowcasting) em favor de um modelo preditivo temporal (forecasting a T+1 e T+2) estritamente segmentado pelo ciclo de vida profissional e pela função tática do atleta, estendendo a estrutura analítica de Khalife et al. (2025) de 9 para 12 categorias combinadas.
* Contribuição Prático-Empírica: A pesquisa fornece uma ferramenta decisória quantitativa, objetiva e replicável para departamentos de scouting, análise de mercado e finanças corporativas de clubes futebolísticos de médio porte. Ao fundamentar as estimativas em dados de campo objetivos do FBref, a metodologia reduz a assimetria informacional nas negociações, neutraliza as heurísticas de ancoragem inerentes às estimativas de multidão e otimiza a alocação de capital em operações de recrutamento e revenda de atletas.
