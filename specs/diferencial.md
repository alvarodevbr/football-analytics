Diferencial do Meu Estudo

1. Introdução e Contextualização do Problema Real

1.1 O Desafio Mercadológico dos Clubes Médios (Modelo 'Buy-Low / Sell-High')

A sustentabilidade financeira no futebol profissional contemporâneo consolidou-se como um desafio estrutural complexo, derivado de um cenário de superendividamento, tomada de decisão pautada no curto prazo e severa inflação no mercado global de transferências (Hill et al., 2025). Esse ecossistema inflacionado impõe restrições severas não apenas a clubes de futebol de médio porte, mas também a desenvolvedores e fornecedores de ferramentas analíticas (analytics solution providers). No ecossistema B2B de dados esportivos, empresas de tecnologia enfrentam elevada volatilidade e insatisfação de clientes (churn) quando seus algoritmos de valoração reproduzem o ruído estocástico e as distorções do mercado, gerando estimativas instáveis ou artificialmente inflacionadas a partir de dados brutos não purificados.

Para mitigar esses riscos e garantir a eficiência alocativa de capital em departamentos de recrutamento com orçamentos restritos, o modelo Buy-Low / Sell-High consolida-se como uma diretriz estratégica vital. Essa abordagem pauta-se na identificação quantitativa de atletas cujos custos de aquisição estão defasados em relação ao seu rendimento tático real, viabilizando o posterior desenvolvimento esportivo e a revenda por margens de lucro substanciais. Essa arquitetura metodológica fundamenta-se na lógica da tomada de decisão orientada a dados (Moneyball), operacionalizada com sucesso por organizações como FC Midtjylland e TSG Hoffenheim, que demonstraram como a modelagem estatística rigorosa permite identificar talentos subavaliados e converter eficiência operacional em ganho patrimonial no balanço corporativo (Müller et al., 2017).

Do ponto de vista da teoria financeira corporativa aplicada ao esporte, o modelo alinha-se estritamente à definição formal de valor proposta por Hill et al. (2025): o desempenho esportivo tático-operacional alcançado em relação ao custo total de aquisição. Atletas que entregam alto rendimento tático a um custo de aquisição modesto situam-se na autêntica "zona de valor". Para responder a essa demanda prática, o presente trabalho direciona seu foco analítico para o segmento dos 90\% mais baratos do mercado de transferências (abaixo do 90^\text{o} percentil). Ao construir um arcabouço preditivo especializado nessa faixa de mercado, a pesquisa fornece a desenvolvedores de software e analistas de clubes um backend algorítmico robusto, capaz de entregar estimativas estáveis, reproduzíveis e imunes a distorções do mercado especulativo.

O modelo Buy-Low / Sell-High desdobra-se em três pilares estratégicos fundamentais:

* Aquisição em Zona de Valor (Buy-Low): Mapeamento sistemático de atletas cujo preço de negociação se encontra abaixo do seu valor relativo intrínseco, mensurado via vetores multidimensionais de desempenho tático-operacional em campo.
* Maximização da Eficiência Esportiva: Aproveitamento da produtividade tática do atleta durante a vigência contratual, otimizando o retorno sobre o investimento em vitórias, pontos acumulados e premiações operacionais.
* Desinvestimento Estratégico (Sell-High): Monetização do ativo no ápice de sua curva de valoração de mercado, gerando superávits financeiros e caixa líquido para reinvestimento no ciclo contínuo de recrutamento do clube.

1.2 A Distorção das Superestrelas e a Conflatação entre Preço e Valor

A literatura econométrica e as ferramentas tradicionais de precificação frequentemente falham ao tratar o mercado de transferências como um ambiente homogêneo, incorrendo no viés teórico de conflatar o preço transacionado de um atleta com o seu valor intrínseco. Evidências empíricas consolidadas por Müller et al. (2017) demonstram que modelos estatísticos perdem capacidade preditiva e precisão paramétrica ao tentar valorar o topo da distribuição do mercado — especificamente o segmento correspondente ao top 10\% mais caro (transações situadas acima do 90^\text{o} percentil ou €18\text{ milhões}). Nesse estrato de superestrelas, as cotações financeiras são severamente inflacionadas por componentes intangíveis extracampo, tais como apelo comercial, brand equity, visibilidade midiática global e potencial de venda de licenciamentos, desalinhando o preço final do rendimento estritamente tático.

Para superar essa assimetria teórica, este trabalho operacionaliza a distinção conceitual rigorosa estabelecida por Hill et al. (2025), adaptando a tipologia de valoração corporativa de Aswath Damodaran ao mercado do futebol. Enquanto os modelos de Fluxo de Caixa Descontado (DCF/Intrinsic Valuation) enfrentam limitações severas no futebol devido à impossibilidade de isolar fluxos de caixa individuais de um único atleta e à ausência de valor residual ao fim do contrato pós-Lei Bosman, o presente estudo apoia-se nas tipologias de Valoração Relativa e Valoração Probabilística via Machine Learning (Hill et al., 2025). Sob esse arcabouço, estabelece-se o desacoplamento conceitual estrito entre Preço e Valor.

Definição Formal de Preço vs. Valor no Futebol Profissional (Hill et al., 2025):

* Preço: O montante financeiro negociado e efetivamente pago entre entidades desportivas em um evento de transferência. Trata-se de uma variável exógena de mercado determinada por poder de barganha relativo, tempo restante de contrato, capacidade de liquidez do clube comprador, necessidade tática imediata, restrições regulatórias de Financial Fair Play e estratégias de negociação de intermediários.
* Valor: A estimativa probabilística do benefício econômico-esportivo intrínseco gerado pelo atleta. Deriva exclusivamente do rendimento tático-operacional em campo (gols, passes, interceptações, duelos vencidos) ponderado pelas características físicas e estruturais do jogador (posição e idade).

Conforme estipulado no mandato teórico de Hill et al. (2025), variáveis como o tempo restante de contrato, a urgência de negociação e a saúde financeira dos clubes compradores e vendedores atuam estritamente como determinantes de Preço, não devendo integrar os vetores de mensuração de Valor. O valor intrínseco de um atleta é puramente uma função de sua produtividade físico-tática em campo e de seus atributos biométricos e posicionais.

2. Pilares Metodológicos e Inovações do TCC

2.1 Purificação da Amostra e Filtros Metodológicos Rigorosos

A consecução de modelos preditivos com elevada capacidade de generalização exige um protocolo estrito de limpeza e purificação da base de dados, desenhado para eliminar ruídos estocásticos decorrentes do viés de marca e variações de extrema variância.

1. Filtro: Exclusão do Top 10\% de Superestrelas (Corte no 90^\text{o} Percentil / €18\text{ Milhões})
  * Critério Técnico: Remoção de todas as observações e transações situadas no estrato superior do 90^\text{o} percentil da distribuição de mercado (€18\text{ milhões}).
  * Justificativa Científica: Purificação amostral para eliminar o viés de marca intangível (brand equity) e atributos comerciais que distorcem as cotações do topo do mercado, nicho onde os algoritmos preditivos tradicionais perdem poder de explicação (Müller et al., 2017).
2. Filtro: Minutagem Mínima em Campo (\ge 90 Minutos por Temporada)
  * Critério Técnico: Exclusão sistemática de atletas com participação acumulada inferior a 90 minutos de jogo na temporada analisada.
  * Justificativa Científica: Mitigação do viés de pequenas amostras e estabilização da variância estatística dos vetores de desempenho tático-operacional, assegurando representatividade à métrica do atleta (Brandes & Franck, 2012; Müller et al., 2017).
3. Filtro: Remoção Completa da Posição de Goleiros
  * Critério Técnico: Exclusão da amostra de todos os atletas que atuam especificamente na posição de goleiro.
  * Justificativa Científica: As dinâmicas táticas, ações motoras e métricas de desempenho de goleiros diferem ontologicamente das de atletas de linha, exigindo uma matriz de dados e um modelo de valoração totalmente isolado e independente (Bryson et al., 2012; Lucifora & Simmons, 2003; Müller et al., 2017).

2.2 Isolamento Tático e Controle Contratual Multifonte (FBref vs. Transfermarkt)

A arquitetura de engenharia de dados do TCC estabelece um desacoplamento multifonte entre métricas de desempenho físico-tático e variáveis financeiro-contratuais. Essa estratégia atende diretamente às objeções da literatura recente (Coates & Parshakov, 2022; Hill et al., 2025) no tocante aos vieses de percepção e cascatas de informação inerentes a plataformas crowdsourced.

Ao restringir o uso dos dados do Transfermarkt estritamente a controles contratuais e financeiros, a metodologia contorna distorções cognitivas associadas à "sabedoria das multidões", tais como o viés de ancoragem (Tversky & Kahneman, 1974) e a dependência de estimativas anteriores. Simultaneamente, o vetor de valoração tática é construído de forma limpa a partir de provedores de dados objetivos de partida (FBref / Opta / Wyscout).

Fonte de Dados & Tipo de Variável	Função Metodológica no Modelo Preditivo
FBref / Opta / Wyscout<br>(Métricas Táticas e Desempenho em Campo)	Captura os vetores de rendimento tático puro (passes certos, percentual de acerto de passe, dribles bem-sucedidos, desarmes, duelos aéreos vencidos, interceptações, gols e assistências). Atua como a variável independente base para a determinação do valor tático intrínseco.
Transfermarkt<br>(Métricas Financeiras e Contratuais)	Utilizado estritamente para controle financeiro, histórico de cotações de mercado e dados cadastrais/contratuais (tempo restante de contrato, histórico de transferências). Os dados biométricos e cadastrais (idade) atuam como variáveis de controle de ciclo de vida, isolando fatores de preço sem contaminar o vetor de desempenho tático.

2.3 Validação Granular Estratificada em 12 Subgrupos (Idade x Posição)

A inovação central deste trabalho frente à literatura recente consiste na implementação de uma matriz de validação cruzada estratificada em 12 subgrupos combinados (4 \times 3). Pesquisas precedentes no campo do aprendizado de máquina adotaram abordagens altamente agregadas ou matrizes reduzidas — como o estudo de Khalife et al. (2025), que utilizou uma segmentação de 9 grupos (3 \times 3: faixas etárias de 16\text{--}23, 24\text{--}29, e 30+ anos para 3 posições).

A necessidade teórica e matemática de expandir a resolução da matriz para um formato 4 \times 3 sustenta-se diretamente nos achados empíricos de Khalife et al. (2025) e Hill et al. (2025):

1. Impacto Preditivo da Variável Potencial: Khalife et al. (2025) provaram que a inclusão do potencial futuro projetado eleva a precisão do modelo preditivo em Gradient Boosting de uma média de 74\% para 84\% (R^2), mantendo o erro quadrático médio normalizado (RMSE) entre 2\% e 5\%.
2. Dinamismo e Decaimento de Atributos: A relevância das features de desempenho altera-se radicalmente ao longo da carreira. Atributos como finalização dominam para atacantes, passe e visão para meio-campistas, e desarmes/velocidade para defensores.
3. Quebra Preditiva na Fase Veterana em Matrizes 3 \times 3: No estudo de Khalife et al. (2025), o modelo de 9 subgrupos sofreu uma queda acentuada de precisão na coorte de atacantes acima de 30 anos (com o R^2 caindo para 59\%). Essa degradação ocorre porque a faixa etária 30+ mescla atletas no topo da maturidade tática (28\text{--}32 anos) com atletas em franca depreciação físico-motora (32+ anos).

Ao subdividir a carreira do atleta em 4 faixas etárias e cruzá-las com 3 posições de linha, o TCC elimina essa perda de precisão preditiva nos atletas veteranos, capturando adequadamente a curva não-linear de depreciação físico-tática.

Faixa Etária / Posição Tática	Defensor	Meio-Campista	Atacante
16-23 anos (Desenvolvimento / Alto Potencial)	Defensor Jovem	Meio-Campista Jovem	Atacante Jovem
24-28 anos (Pico de Desempenho / Ápice)	Defensor em Ápice	Meio-Campista em Ápice	Atacante em Ápice
28-32 anos (Maturidade Tática)	Defensor Maturo	Meio-Campista Maturo	Atacante Maturo
32+ anos (Fase Veterana)	Defensor Veterano	Meio-Campista Veterano	Atacante Veterano

3. Quadro Comparativo e Contribuições do Estudo

3.1 Quadro Comparativo de Contribuições frente à Literatura de Referência

A tabela a seguir sistematiza as contribuições e os avanços metodológicos da presente pesquisa em relação aos marcos teóricos da literatura de valoração desportiva.

Estudo / Referência	Abordagem Metodológica	Limitação Observada	Contribuição Diferencial do TCC
Müller et al. (2017)	Regressão multinível aplicada a 5 ligas europeias em 6 temporadas.	Identificou a perda de precisão no top 10\% caro, mas sem segmentação contínua cruzando faixa etária e posição.	Purifica a amostra limitando-se ao segmento dos 90\% mais baratos e aplica validação estratificada em 12 subgrupos.
Coates & Parshakov (2022)	Análise empírica do viés preditivo e da "sabedoria das multidões" no Transfermarkt.	Demonstrou os vieses de ancoragem e cascatas de informação no crowdsourcing, sem formular um algoritmo substituto de valoração tática.	Desacopla as fontes de dados: restringe o Transfermarkt a variáveis de preço/contrato e utiliza o FBref/Opta para valoração tática objetiva.
van Arem et al. / Abordagens Tradicionais	Modelos econométricos agregados de regressão linear MCO.	Tratam o mercado como um bloco homogêneo, ignorando as variações não-lineares da depreciação físico-tática.	Estrutura modelos preditivos isolados para cada um dos 12 subgrupos etários e posicionais, eliminando o viés de agregação.
Khalife et al. (2025)	Gradient Boosting Machine Learning segmentado em 9 subgrupos (3 \times 3).	A matriz 3 \times 3 sofreu queda acentuada de precisão em atacantes veteranos (R^2 de 59\%), agrupando fases heterogêneas de carreira.	Expande a resolução preditiva para uma matriz 4 \times 3 (12 subgrupos), isolando a fase de maturidade (28\text{--}32) do estrato veterano (32+) e estabilizando a acurácia.
Hill et al. (2025)	Revisão conceitual e alinhamento das tipologias de Damodaran ao futebol.	Estudo essencialmente teórico e qualitativo, carente de implementação e validação algorítmica em bases de dados empíricas.	Operacionaliza as tipologias de Valoração Relativa e Probabilística de Damodaran via algoritmos de Gradient Boosting treinados com dados reais.

3.2 Detalhamento das Contribuições Científicas e Aplicações Práticas

Contribuição Científica
A pesquisa consolida um arcabouço preditivo de elevada precisão para o campo das finanças corporativas no futebol e do Sport Analytics. Ao isolar o ruído estocástico das superestrelas (top 10\% do mercado) e operacionalizar as tipologias de valoração relativa e probabilística de Damodaran (Hill et al., 2025), o estudo estabelece um padrão metodológico rigoroso para a precificação de ativos esportivos. A implementação da matriz estratificada 4 \times 3 soluciona empiricamente as falhas de acurácia reportadas em modelos de menor resolução (como a quebra preditiva em coortes veteranas observada por Khalife et al., 2025), oferecendo uma modelagem não-paramétrica por árvore de decisão (Gradient Boosting) capaz de capturar variações dinâmicas de atributos ao longo de todo o ciclo de vida do atleta.

Aplicação Prática (Mercado & Clubes Médios)
Para o mercado operacional de recrutamento e desenvolvedores de ferramentas analíticas, o estudo entrega um algoritmo robusto e de baixo custo computacional para identificação de assimetrias de precificação (pricing mispricing). Ao isolar variáveis de preço (como tempo contratual restante) dos vetores de valor (desempenho tático em campo), a ferramenta permite que clubes de médio porte identifiquem atletas subavaliados — cujo valor tático estimado supera o preço pedido no mercado negociado —, maximizando a eficiência alocativa de capital e otimizando o retorno sobre o investimento (ROI) em operações de transferência.
