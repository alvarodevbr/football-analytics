5. Verificação dos dados

1. Arquitetura da Base de Dados e Fontes de Informação

A construção de um arcabouço econométrico robusto para a valoração e predição de desempenho de jogadores de futebol exige a integração multissistema de bases de dados heterogêneas. A opacidade histórica do mercado de transferências europeu e a ausência de um repositório financeiro público unificado requerem o acoplamento de fontes especializadas em estatísticas tático-técnicas a plataformas consolidadas na estimativa de ativos e dados contratuais (Hill et al., 2025; Simmons, 2021).

Neste trabalho, a arquitetura de dados é estruturada a partir da fusão sistemática de duas fontes primárias, selecionadas por critérios de consistência metodológica e abrangência empírica:

1. FBref (alimentado pelos provedores de dados de rastreamento de eventos StatsBomb e Opta): Constitui a fonte primária para o vetor de variáveis explicativas de desempenho tático-técnico em campo. A opção pelo FBref fundamenta-se na disponibilidade de métricas padronizadas, publicamente auditáveis e desagregadas ao nível de evento individual por partida, cobrindo volume e precisão de passes, finalizações, conduções progressivas, ações defensivas, duelos aéreos e disciplina.
2. Transfermarkt: Atua como fonte de referência para os valores de mercado estimados (V), dados contratuais (duração restante do contrato e data de término) e características demográficas e físicas dos atletas (idade, posição primária, nacionalidade, pé preferencial e altura). A plataforma é reconhecida na literatura como a proxy padrão para valoração relativa na ausência de preços contínuos de transação (Müller et al., 2017; Herm et al., 2014).

O recorte temporal e geográfico compreende o acompanhamento longitudinal dos atletas atuantes nas cinco principais ligas do continente europeu (Big Five): Premier League (Inglaterra), La Liga (Espanha), Bundesliga (Alemanha), Serie A (Itália) e Ligue 1 (França). A janela de observação abrange o ciclo de 5 temporadas consecutivas — de 2019/20 a 2023/24 —, garantindo uma amostragem de painel balanceada e representativa das dinâmicas financeiras e esportivas contemporâneas.

A Tabela 5.1 sintetiza a estrutura metodológica e o papel analítico de cada fonte na modelagem econométrica.

Tabela 5.1: Matriz comparativa das fontes primárias de dados.

Fonte de Dados	Categoria de Variáveis Extraídas	Propósito Metodológico no Modelo
FBref	Métricas tático-técnicas de desempenho em campo (passes, finalizações, desarmes, conduções, ações defensivas e minutagem).	Mensuração do desempenho produtivo ex-ante do atleta e vetor de produtividade tática em T.
Transfermarkt	Valores de mercado estimados (V), dados contratuais (duração, término) e atributos demográficos/físicos (idade, posição, nacionalidade, pé, altura).	Definição do baseline financeiro do ativo, controle demográfico/contratual e construção das variáveis-alvo de variação (\Delta V e \Delta V\%).

2. Critérios de Elegibilidade e Recorte Metodológico da Amostra

A definição da amostra de trabalho seguiu um funil de seleção sequencial rigoroso, concebido para mitigar vieses de seleção, reduzir ruídos de pequenas amostras e eliminar distorções provocadas por observações atípicas (outliers) em segmentos de mercado não regidos pela produtividade técnica tradicional.

O processo de depuração amostral é detalhado a seguir:

1. Amostra Bruta Multitemporal: Consolidação inicial de todos os registros de atletas inscritos em clubes das ligas do Big Five entre as temporadas 2019/20 e 2023/24 (N_{bruto} \approx 12.500 observações-atleta/temporada).
2. Aplicação do Filtro do Percentil 90 de Valor de Mercado (Filtro p90): Restrição da amostra aos 90% atletas de menor valor de mercado em T, mediante a exclusão do topo 10% mais valioso da distribuição. O corte empírico fundamenta-se nos achados de Müller et al. (2017), que estabelecem que modelos estatístico-regressivos e algoritmos de aprendizado de máquina possuem elevada precisão preditiva para atletas com valor de mercado de até aproximadamente €18 milhões (limiar empírico do percentil 90 nas ligas europeias). Acima desse patamar, o desempenho dos modelos parametrizados perde eficácia para o crowdsourcing, pois a precificação de superestrelas (ex.: Mbappé, Haaland) atende a lógicas intangíveis de apelo comercial, potencial de marketing, direitos de imagem e comportamento desproporcional de leilão entre clubes com restrição orçamentária branda.
3. Aplicação do Filtro de Minutagem Mínima (90+ Minutos em T): Exigência de que o atleta tenha atuado por no mínimo 90 minutos acumulados na temporada T na base do FBref. Essa restrição elimina o viés de pequenas amostras e a variância extrema decorrente de aparições esporádicas, assegurando que métricas padronizadas por jogo ou por 90 minutos sejam estatisticamente representativas da produtividade do ativo (Brandes & Franck, 2012; Müller et al., 2017).
4. Exclusão Técnica da Posição de Goleiro: Remoção integral dos guardiões de meta da amostra final. A literatura especializada em economia do esporte e ciência de dados aplicada (Lucifora & Simmons, 2003; Bryson et al., 2012; Müller et al., 2017; Khalife et al., 2025) demonstra que goleiros possuem vetores de produtividade e vetores de desempenho tático radicalmente heterogêneos em relação aos jogadores de linha (ex.: defesas difíceis, taxa de defesas, gols sofridos vs. passes progressivos, desarmes, finalizações). A manutenção dessa categoria comprometeria a especificidade dos estimadores ou exigiria uma parametrização paralela desconectada da matriz vetorial dos atletas de campo.
5. Amostra Final Refinada: A base resultante consolida uma estrutura de dados em painel altamente homogênea, otimizada para o treinamento de algoritmos regressivos e preditivos.

3. Matriz de Segmentação em 12 Categorias e Validação Cruzada

Para capturar a heterogeneidade estrutural e o caráter não linear da valoração ao longo do ciclo de vida profissional, aplica-se uma matriz de segmentação 4 \times 3. A estrutura cruza quatro faixas etárias representativas dos estágios de maturação biológica e esportiva com três posições genéricas de linha.

Cabe explicitar uma nuance metodológica fundamental: embora a referência teórica de Khalife et al. (2025) tenha utilizado originalmente uma divisão tripartite do ciclo de vida (16–23 anos: estágio inicial; 24–29 anos: estágio intermediário; 30+ anos: estágio final), este estudo expande a classificação para quatro faixas etárias (16–23, 24–28, 28–32, 32+ anos). Essa adaptação justifica-se pela necessidade de isolar a janela crítica de transição entre o pico absoluto de rendimento físico-técnico (24–28 anos) e a fase de consolidação/início da desaceleração física (28–32 anos), permitindo uma mensuração mais precisa da taxa de depreciação do capital humano.

3.1. Estruturação dos Eixos da Matriz

A matriz organiza-se a partir da combinação dos seguintes eixos:

* Faixas Etárias:
  1. 16–23 anos: Estágio de Desenvolvimento e Prospecção (valoração fortemente ancorada no potencial futuro, expectativa de crescimento e maior volatilidade).
  2. 24–28 anos: Estágio do Pico de Desempenho e Maturidade (estabilidade produtiva, máxima eficiência física e valorização de mercado no ápice).
  3. 28–32 anos: Estágio de Consolidação e Transição (início da desaceleração valormétrica física, com precificação sustentada pela reputação e leitura tática).
  4. 32+ anos: Estágio de Veteranos e Depreciação (curva descendente acentuada do valor de mercado, impulsionada pelo risco biológico e perda de dominância física).
* Posições de Linha: Defensor, Meio-Campista e Atacante.

A Tabela 5.2 detalha a Matriz de 12 Categorias (C1 a C12) e os determinantes de valor associados a cada segmento, em alinhamento com a literatura de finanças esportivas (Khalife et al., 2025).

Tabela 5.2: Matriz de segmentação de ativos em 12 categorias (4 \times 3).

Categoria ID	Faixa Etária	Posição	Perfil e Fatores Determinantes de Valor (Khalife et al., 2025)
C1	16–23 anos	Defensor	Foco em ações defensivas básicas, duelos terrestres e projeção de desenvolvimento físico.
C2	16–23 anos	Meio-Campista	Foco na capacidade de passe, visão de jogo, retenção de bola e potencial de carreira.
C3	16–23 anos	Atacante	Foco na capacidade de finalização, drible, velocidade acelerada e alto potencial futuro.
C4	24–28 anos	Defensor	Maturidade tática, duelos aéreos/terrestres eficientes, posicionamento e qualidade no passe.
C5	24–28 anos	Meio-Campista	Equilíbrio ótimo entre volume de passes, criação de chances, cobertura defensiva e intensidade.
C6	24–28 anos	Atacante	Pico produtivo: conversão de gols, assistências, taxa de sucesso no drible e volume de finalização.
C7	28–32 anos	Defensor	Leitura de jogo antecipatória, posicionamento defensivo e experiência internacional.
C8	28–32 anos	Meio-Campista	Controle de ritmo de jogo, precisão de passes em zonas sob pressão e reputação acumulada.
C9	28–32 anos	Atacante	Eficiência finalizadora, presença em seleções nacionais e inteligência de movimentação.
C10	32+ anos	Defensor	Declínio do ritmo (pace); valoração altamente dependente do histórico, reputação e minutagem.
C11	32+ anos	Meio-Campista	Experiência técnica e refino no passe; forte dependência da minutagem e status de liderança.
C12	32+ anos	Atacante	Acentuada depreciação de valor; impacto severo do fator idade, minutagem e histórico de lesões.

3.2. Estratégia de Validação Cruzada (K-Fold) e Estabilidade dos Estimadores

Para garantir a generalização preditiva dos modelos econométricos e de aprendizado de máquina ajustados dentro de cada uma das 12 categorias, adota-se um protocolo rigoroso de Validação Cruzada K-Fold (K=5 ou K=10), em conformidade com as diretrizes metodológicas de Khalife et al. (2025).

A aplicação da validação cruzada justifica-se pela necessidade de mitigar o risco de overfitting (superajuste), que ocorre quando o modelo memoriza o ruído dos dados de treinamento de um subgrupo específico em detrimento da capacidade preditiva fora da amostra (out-of-sample). No procedimento K-Fold:

1. A amostra de cada uma das 12 categorias é particionada aleatoriamente em K subconjuntos (dobras) de igual tamanho.
2. O modelo é treinado iterativamente em K-1 dobras e testado na dobra remanescente, repetindo-se o processo K vezes até que todos os blocos tenham servido como base de teste.
3. A performance preditiva final é reportada pela média do R^2 Ajustado e pela Raiz do Erro Quadrático Médio (RMSE / RMSE Normalizado).

Essa abordagem garante estabilidade estatística aos coeficientes das variáveis tático-técnicas e demográficas, permitindo avaliar se o poder explicativo das métricas de desempenho varia de forma consistente entre jovens talentos (ex.: C1, C3) e atletas em fase de depreciação (ex.: C10, C12).

4. Mapeamento das Variáveis Explicativas (T) e Variáveis-Alvo (\Delta V e \Delta V\%)

O framework de modelagem adota uma estrutura temporizada ex-ante/ex-post. As variáveis explicativas (vetores tático-técnico, financeiro e demográfico) são mensuradas estritamente no momento inicial T. Em contrapartida, os retornos e variações econômico-financeiras do ativo são observados nos horizontes futuros de um e dois anos (T+1 e T+2).

4.1. Definição das Variáveis-Alvo de Variação

As variáveis dependentes quantificam a variação absoluta e relativa no valor de mercado do atleta nos horizontes k \in \{1, 2\}:

* Variação Absoluta (\Delta V_{T+k}): \Delta V_{T+k} = V_{T+k} - V_T
* Variação Percentual (\Delta V\%_{T+k}): \Delta V\%_{T+k} = \left( \frac{V_{T+k} - V_T}{V_T} \right) \times 100

em que V_T representa o valor de mercado no ano-base mensurado no Transfermarkt e V_{T+k} representa o valor de mercado observado k anos após o término da temporada de medição do rendimento tático.

4.2. Mapeamento dos Vetores Explicativos

Os vetores de variáveis independentes coletados em T são especificados econometricamente conforme descrito abaixo:

* Vetor de Desempenho Tático-Técnico (FBref):
  * Minutos jogados: Volume acumulado na temporada T;
  * Gols marcados e Assistências fornecidas: Métricas brutas e por 90 minutos;
  * Passes tentados e Taxa de acerto de passes (%);
  * Dribles certos e Taxa de sucesso no drible (%);
  * Duelos aéreos vencidos e Taxa de sucesso em duelos aéreos (%);
  * Desarmes efetuados e Interceptações;
  * Clean sheets induzidos: Partidas sem sofrer gols enquanto o atleta esteve em campo (aplicado a defensores e meio-campistas);
  * Indisciplina: Cartões amarelos e vermelhos acumulados.
* Vetor Financeiro e Contratual (Transfermarkt):
  * Valor de mercado base (V_T): Nível financeiro inicial do ativo em T;
  * Duração restante do contrato: Mensurada continuamente em anos até a data de expiração;
  * Status contratual (Âncora Bosman): Variável dummy indicando se o atleta está no último ano de contrato (Status\_Bosman_i = 1; 0 caso contrário), capturando o risco de saída por transferência livre.
* Vetor Demográfico e Intangível:
  * Idade do atleta em T e o termo quadrático Idade^2: Especificado formalmente para capturar a concavidade da curva de rendimentos marginais do capital humano ao longo do ciclo de vida;
  * Altura (cm) e Pé preferencial (variável indicadora para atletas ambidestros);
  * Nacionalidade / Continente de origem: Vetor de variáveis dummy regionais (UEFA_i, CONMEBOL_i, CAF_i, AFC/CONCACAF_i) para controlar potenciais prêmios de mercado ou dinâmicas de passaporte comunitário;
  * Convocação para Seleção Nacional: Variável dummy indicadora (Seleção_i \in \{0, 1\}), assumindo valor 1 se o atleta foi convocado para a seleção principal em T;
  * Reputação Internacional: Índice quantificado ordinal do prestígio global do atleta em plataformas especializadas.

5. Análise Crítica das Limitações Metodológicas da Base de Dados

5.1. Vieses Estruturais do Transfermarkt e a Falha da "Sabedoria das Massas"

Embora o Transfermarkt seja amplamente empregado em pesquisas empíricas como proxy para o valor dos futebolistas, sua estrutura de geração de dados apresenta fragilidades conceituais quando confrontada com a teoria financeira e com os fundamentos da inteligência coletiva (Hill et al., 2025; Coates & Parshakov, 2022; Herm et al., 2014).

A literatura frequentemente assume que os valores do Transfermarkt decorrem do fenômeno da "Sabedoria das Massas" (Wisdom of Crowds), formulado por Francis Galton e sistematizado por Surowiecki (2004). Contudo, Hill et al. (2025) e Coates & Parshakov (2022) demonstram que, para que a sabedoria das massas produza estimativas não viesadas, cinco condições obrigatórias devem ser satisfeitas: (1) Diversidade de opiniões, (2) Independência dos julgamentos, (3) Descentralização, (4) Agregação adequada e (5) Confiança.

A arquitetura do Transfermarkt viola abertamente pelo menos três desses princípios básicos:

1. Violação da Independência: Os usuários do fórum têm acesso irrestrito às estimativas prévias salvas no sistema e às discussões de outros membros. A falta de cegueira nos palpites desencadeia o viés de Ancoragem e Ajuste (Tversky & Kahneman, 1974), em que a estimativa inicial ou o valor de mercado vigente atua como uma âncora cognitiva, da qual os avaliadores fazem apenas ajustes marginais insuficientes. Adicionalmente, gera Cascatas de Informação (Bikhchandani et al., 1992), onde usuários posteriores ignoram seus sinais privados e alinham seus votos ao consenso emergente ou a especulações da imprensa.
2. Violação da Descentralização e Agregação (O Princípio do Juiz): Conforme revelado por Herm et al. (2014), os valores finais do Transfermarkt não resultam da média ou mediana estatística dos palpites da comunidade. A plataforma adota o Judge Principle ("Princípio do Juiz"), em que administradores regionais selecionam, ponderam, alteram ou descartam discricionariamente as estimativas dos usuários. Esse filtro centralizador introduz uma camada não transparente de julgamento subjetivo.

Em suma, embora as estimativas do Transfermarkt apresentem alta correlação com transações reais, elas contêm vieses comportamentais estruturais que devem ser considerados na interpretação dos resíduos dos modelos econométricos.

5.2. Desafios Teórico-Práticos de Coleta, Scraping e Fusão de Bases (ID Matching)

A consolidação do banco de dados unificado exigiu procedimentos avançados de engenharia de dados para a fusão das bases do FBref e do Transfermarkt. O obstáculo metodológico central reside na inexistência de uma chave primária universal (Unique Player ID) compartilhado entre as duas plataformas.

Para operacionalizar o pareamento dos registros, desenvolveram-se algoritmos de correspondência aproximada de cadeias de caracteres (fuzzy string matching), combinando distância de Levenshtein e similaridade de Jaro-Winkler, complementados por limpeza e validação manual. O processo enfrentou e solucionou três categorias de ruído:

1. Variações Grafométricas: Discrepâncias no registro de nomes próprios entre fontes distintas (ex.: uso do nome completo de batismo em uma plataforma versus apelidos ou prenomes abreviados na outra).
2. Codificação de Caracteres Especiais (Encoding): Incompatibilidades no padrão UTF-8 associadas a alfabetos não latinos ou acentuações regionais (caracteres cirílicos, nórdicos, turcos ou eslavos).
3. Homônimos: Falsos positivos gerados por atletas com nomes idênticos atuando na mesma liga. O problema exigiu a imposição de condicionantes multivariados de checagem cruzada, validando simultaneamente data de nascimento, altura, nacionalidade e clube de vínculo em T.

5.3. A Opacidade Financeira: Distinção Conceitual entre Preço e Valor e Cláusulas Ocultas

A limitação teórica e empírica mais severa na modelagem de ativos no futebol profissional reside na opacidade contratual e na confusão conceitual entre Preço de Mercado e Valor Inerente/Intrínseco (Damodaran, 2012, 2018; Hill et al., 2025).

Apoiando-se nos arcabouços de finanças corporativas aplicados ao esporte (Damodaran, 2012; Hill et al., 2025), estabelece-se a seguinte distinção analítica:

* Valor Inerente (Intrínseco / Produtivo): Refere-se à capacidade do atleta de gerar utilidade e retorno esportivo no campo de jogo. É determinado exclusivamente pelo vetor de desempenho tático-técnico (métricas extraídas do FBref), idade, experiência e posição do jogador.
* Preço de Aquisição (Preço de Mercado): Refere-se à quantia financeira efetivamente acordada em uma transação de transferência. O preço é profundamente influenciado por fatores exógenos ao atleta, tais como: liquidez financeira do comprador, necessidade urgente de reposição de elenco, poder de barganha do vendedor, tempo restante de contrato, restrições do Fair Play Financeiro da UEFA e comissões de agentes.

No futebol europeu, ao contrário do esporte corporativo norte-americano (MLB, NBA, MLS) — onde os dados salariais e contratuais são públicos por exigência regulatória —, opera-se sob estrita opacidade financeira (Simmons, 2021). A ausência de dados salariais oficiais força a pesquisa acadêmica a recorrer a proxies (como Capology ou estimativas de imprensa).

Além disso, variáveis contratuais cruciais permanecem inacessíveis e não observáveis para o econometrista:

* Estruturas de bônus por metas contingentes de desempenho individual e coletivo;
* Percentuais retidos sobre direitos de imagem e comissões pagas a intermediários;
* Cláusulas de rescisão (buy-out clauses) restritivas ou preferências ocultas de recompra.

A impossibilidade de observar empiricamente essas condicionantes contratuais privadas implica que uma parcela da variância do mercado não pode ser capturada pelas métricas tático-técnicas do FBref ou demográficas do Transfermarkt. Essa limitação estrutural de variância não explicada é um limite metodológico intrínseco aos modelos de valoração no futebol europeu, devendo ser devidamente contextualizada na análise dos resultados empíricos.
