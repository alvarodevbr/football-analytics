Definição da Metodologia

1. Visão Geral e Contexto Operacional

O futebol profissional europeu contemporâneo opera sob uma dinâmica industrial refinada e complexa, caracterizada pela escalada inflacionária nas taxas de transferência, pelo rigor normativo de regras de sustentabilidade financeira (como o Financial Fair Play da UEFA e o Independent Football Regulator) e por assimetrias estruturais de capital. Nesse ambiente, clubes de médio porte e organizações focadas na revelação e no desenvolvimento de ativos encontram-se expostos a margens de erro financeiro extremamente exíguas. Para esses atores, a sustentabilidade institucional e a competitividade desportiva dependem criticamente da execução otimizada do modelo de negócios comercial do tipo buy-low / sell-high — a identificação e aquisição sistemática de atletas subavaliados pelo mercado, seu desenvolvimento tático-técnico e posterior alienação com expressivo ágio financeiro. A formalização de uma metodologia quantitativa e preditiva rigorosa busca substituir intuições empíricas por um processo decisório orientado por ciência de dados, alocando capital de forma eficiente e mitigando os riscos inerentes à transação de capital humano no esporte.

Do ponto de vista da teoria financeira aplicada, a racionalidade do modelo buy-low / sell-high é fundamentada na conceituação do valor do ativo futebolístico. Alinhando a literatura corporativa aos desenvolvimentos recentes das finanças do esporte (Damodaran, 2012, 2018; Follert & Gleißner, 2024; Hill et al., 2025), o valor de um atleta é formalmente definido como o retorno de desempenho desportivo alcançado em relação ao seu custo de aquisição (performance attained relative to acquisition cost). Operacionalmente, esta pesquisa enquadra a arquitetura analítica proposta como um Modelo de Valoração Híbrido Relativo-Probabilístico (Hybrid Relative-Probabilistic Valuation Framework), integrando a taxonomia de valoração corporativa de Damodaran conforme sintetizada por Hill et al. (2025):

1. Dimensão Relativa: O modelo padroniza métricas de desempenho tático-técnico /90 minutos em cohorts comparáveis de atletas (12 subgrupos estratificados), dispensando a necessidade de estimar fluxos de caixa individuais intrínsecos, os quais são praticamente inobserváveis no âmbito individual do futebol.
2. Dimensão Probabilística: O modelo utiliza algoritmos não-paramétricos de aprendizado de máquina (Tree Ensembles) para projetar a distribuição probabilística da realização de valor futuro (\Delta V\%) sob cenários de incerteza de mercado.

Dimensão Operacional	Abordagem Tradicional (Heurística/Empírica)	Abordagem Metodológica Proposta (Data-Driven)
Taxonomia de Valoração	Avaliação intuitiva desprovida de rigor financeiro formal; fusão conceitual entre preço e valor.	Modelo Híbrido Relativo-Probabilístico (Damodaran; Hill et al., 2025): padronização /90 por cohorts e projeção probabilística de ROI.
Avaliação de Talentos	Observação subjetiva de scouts, intermediação de agentes e viés de amostragem em momentos pontuais.	Análise estatística não-linear de vetores tático-técnicos /90 integrados a atributos contratuais e biológicos.
Arbitragem de Mercado	Exposição a viéses cognitivos de ancoragem, encadeamentos informacionais e efeito holofote.	Supressão de assimetrias informacionais no sub-mercado p_{90} via mensuração objetiva do ROI futuro (\Delta V\%).
Gestão do Risco de Aquisição	Pauta-se na reputação histórica do atleta e em amostragem qualitativa suscetível a erros de observação.	Filtragem atuarial (p_{90}, minutagem mínima) e modelagem particionada por matriz etária e posicional (4 \times 3).
Horizonte Preditivo	Análise ex-post do rendimento passado para justificativa do preço negociado na transação.	Validação por Janela Expandida (Expanding Window) focada na projeção do valor intrínseco/relativo em T+1 e T+2.

Tabela 1.1: Alinhamento Estratégico do Modelo de Negócios e Requisitos Metodológicos

A oportunidade de arbitragem financeira para clubes de médio porte reside na ineficiência parcial do mercado de transferências. Enquanto o topo da pirâmide do futebol — os 10% de atletas de valor hiper-elevado (>p_{90}) — opera sob forte visibilidade, a precificação nesse segmento é frequentemente distorcida por fatores não-econômicos (como equidade de marca comercial, sinalização geopolítica de novos acionistas e guerras de ofertas entre superclubes). Em contrapartida, o sub-mercado correspondente aos 90% restantes de atletas de valor médio e baixo sofre com uma severa opacidade de cobertura analítica e viéses heurísticos de scouting (ancoragem, viés de disponibilidade e viés de recenticidade). Essa opacidade cria um desalinhamento estrutural entre o preço praticado e o valor intrínseco/relativo do atleta. Ao substituir heurísticas humanas por mensurações estatísticas objetivas do rendimento físico e tático-técnico, clubes desenvolvedores conseguem extrair alpha financeiro constante, adquirindo ativos subavaliados para revenda com expressivo Retorno sobre o Investimento (ROI).

A viabilização operacional desse modelo preditivo exige uma arquitetura rigorosa de engenharia de dados, responsável pela extração, higienização, harmonização e fusão de bases empíricas heterogêneas.

2. Tratamento, Coleta e Higienização de Dados (Data Pipeline)

O pipeline de dados é a espinha dorsal da arquitetura analítica. Em econometria aplicada ao esporte e modelagem preditiva avançada, a integridade do dado bruto condiciona a validade estatística e a capacidade de generalização dos modelos de aprendizado de máquina. Registros contendo inconsistências de nomenclatura, erros de medição tática ou ruídos decorrentes de amostras não representativas corrompem a superfície de perda dos algoritmos, introduzindo viéses sistemáticos nas estimativas de valoração.

Para assegurar a robustez empírica, estruturou-se um pipeline de extração automatizada via web scraping e modelagem de dataframes consolidados referentes às "Big Five" ligas europeias (Premier League, La Liga, Bundesliga, Serie A e Ligue 1), cobrindo o quinquênio temporal de 2019/20 a 2023/24. A infraestrutura de dados integra duas fontes primárias com papéis complementares:

1. FBref: Fonte primária de métricas tático-técnicas detalhadas de campo, viabilizando a vetorização refinada de eventos por 90 minutos de jogo (/90).
2. Transfermarkt: Plataforma de referência global utilizada para a extração de dados contratuais, idade, posição, nacionalidade e estimativas de valor de mercado dos atletas, consagradas na literatura acadêmica como proxies válidas de valor financeiro (Müller et al., 2017; Prockl & Frick, 2018).

A unificação dessas bases heterogêneas demandou um algoritmo sofisticado de Resolução de Entidades (Entity Resolution). Diante de divergências sistemáticas na grafia dos nomes dos atletas entre o FBref e o Transfermarkt (caracteres diacríticos, nomes compostos, abreviações e variações transliteradas), aplicou-se a correspondência difusa de strings (fuzzy matching) baseada na Distância de Levenshtein. O algoritmo calcula o número mínimo de edições de caractere único (inserções, deleções ou substituições) necessárias para transformar uma string na outra, estabelecendo um limiar de similaridade para fusão autônoma. Nas zonas de ambiguidade do algoritmo, implementou-se um rigoroso protocolo de validação manual, cruzando dados secundários determinísticos (data de nascimento, nacionalidade e histórico de clubes) para erradicar pareamentos falsos (false positives).

Crítico para a fundamentação teórica desta etapa é o entendimento dos mecanismos de precificação do Transfermarkt. A literatura frequentemente assume que os valores da plataforma resultam da "Sabedoria das Multidões" (Wisdom of Crowds). No entanto, conforme demonstrado por Herm et al. (2014) e ressaltado na crítica de Müller et al. (2017), o Transfermarkt opera sob o "Princípio do Juiz" (Judge Principle). Os valores finais de mercado não são gerados por médias democráticas ou simples agregados da comunidade; em vez disso, "juízes" comunitários hierarquicamente nomeados filtram, ponderam e alteram arbitrariamente as propostas dos usuários. Por ser dependente de uma curadoria humana subjetiva, não-transparente e não-reproduzível, a métrica do Transfermarkt exibe vulnerabilidade a viéses de ancoragem e cascatas de informação (Coates & Parshakov, 2022; Hill et al., 2025). Essa constatação teórica fortalece a justificativa deste estudo: substituir a curadoria heurística dos juízes por estimadores algorítmicos puramente orientados a dados (data-driven).

A purificação da base integrada seguiu três critérios estritos de corte atuarial e higienização metodológica:

* Filtro do Percentil 90 (p_{90} - Exclusão de Superestrelas): Eliminou-se o top 10% dos atletas de maior valor absoluto de mercado. Amparando-se nas evidências empíricas de Müller et al. (2017), demonstra-se que a precificação das superestrelas no topo da pirâmide é distorcida por fatores não-lineares e intangíveis (apelo comercial, tração de imagem e poder negocial geopolítico). Ademais, Müller et al. (2017) provaram que modelos estatísticos e de machine learning superam a acurácia das estimativas humanas (crowd/judge estimates) precisamente na faixa amostral referente aos 90% dos atletas de valor baixo e médio — o exato universo operacional dos clubes de médio porte e desenvolvedores.
* Filtro de Minutagem Mínima (\ge 90 minutos em T): Exigiu-se a participação em campo por no mínimo 90 minutos acumulados na temporada de referência T no FBref. Fundamentada nos achados de Brandes & Franck (2012) e Müller et al. (2017), essa restrição garante a representatividade estatística das métricas /90, eliminando artefatos de variância extrema causados por amostras insignificantes (atletas com pouquíssimos minutos jogados que apresentam taxas atipicamente inflacionadas ou nulas).
* Exclusão Técnica de Goleiros: A posição de goleiro foi removida integralmente da amostra. Conforme estabelecido por Berri et al. (2023) e Müller et al. (2017), a especificidade tático-funcional e a disparidade absoluta do espaço de atributos dos goleiros em relação aos atletas de linha inviabilizam a padronização e harmonização de um vetor genérico de features.

Atrição do Data Pipeline e Protocolo de Tratamento de Dados

A movimentação volumétrica e as transformações algébricas do pipeline são formalizadas nas etapas a seguir:

1. Universo Bruto Extraído (N_0 \approx 21.200 registros/temporada-atleta): Coleta dos dados tático-técnicos (FBref) e contratuais/mercadológicos (Transfermarkt) das Big Five ligas europeias (2019/20 a 2023/24).
2. Resolução de Entidades via Levenshtein (N_1 \approx 18.500 registros): Unificação de registros cruzados através da distância de Levenshtein e higienização manual de falsos positivos.
3. Exclusão de Goleiros (N_2 \approx 16.800 registros): Remoção completa da posição de goleiro devido à incomensurabilidade das métricas funcionais (Berri et al., 2023).
4. Filtro de Minutagem Mínima (\ge 90 min em T) (N_3 \approx 12.400 registros): Descarte de observações de baixa representatividade amostral para estancar a volatilidade de métricas /90 (Brandes & Franck, 2012).
5. Corte do Percentil 90 (p_{90}) (N_{final} = 10.350 observações temporada-atleta; N = 4.217 atletas únicos): Supressão do top 10% dos ativos mais caros para isolar a sub-amostra sub-superestrela e otimizar a acurácia do modelo preditivo (Müller et al., 2017).
6. Protocolo de Transformação e Escalonamento de Dados:
  * Transformação Logarítmica \log(1+x): Aplicada à variável de valor de mercado e às variáveis de popularidade/métricas com distribuição fortemente assimétrica à direita (skewed), estabilizando a variância e mitigando o impacto de caldas longas no erro quadrático (Müller et al., 2017).
  * Padronização Robusta (RobustScaler): As variáveis numéricas contínuas /90 foram padronizadas subtraindo-se a mediana e dividindo-se pelo Intervalo Interquartil (IQR = Q_3 - Q_1). Essa técnica é superior ao StandardScaler tradicional pois preserva outliers táticos legítimos (ex.: uma taxa de acerto de passes curtos excepcionalmente alta) sem distorcer o centro da distribuição.

A base purificada e padronizada exige uma estrutura analítica de segmentação tático-posicional e etária capaz de capturar a heterogeneidade da depreciação do capital humano ao longo da carreira dos atletas.

3. Matriz de Segmentação Categoria-Posição (Matriz 4 \times 3)

Modelos preditivos globais aplicados ao futebol falham sistematicamente por assumirem que a dinâmica de valoração é homogênea entre todos os atletas. Na realidade desportiva, os determinantes de valor são altamente dependentes do contexto: o peso de um atributo físico ou técnico flutua substancialmente conforme a função tática ocupada em campo e a fase da trajetória profissional do ativo. Conforme demonstrado empiricamente por Khalife et al. (2025), a tentativa de estimar uma equação única para o universo completo de jogadores introduz um viés severo e reduz o poder preditivo, pois as interações entre as features possuem direções e magnitudes opostas dependendo do papel desempenhado.

Para solucionar essa limitação, esta metodologia expande a abordagem de Khalife et al. (2025) e adota uma Matriz de Segmentação Categoria-Posição de dimensão 4 \times 3, estruturando 12 subgrupos independentes de modelagem preditiva. O espaço total de dados limpos \mathcal{D} é formalmente particionado em 12 subconjuntos mutuamente disjuntos:

\mathcal{D} = \bigcup_{m=1}^{12} \mathcal{D}_m, \quad \text{onde } \mathcal{D}_m \cap \mathcal{D}_k = \emptyset \text{ para } m \neq k

A estimação é formulada independentemente para cada subgrupo m:

Y_{i, T+k}^{(m)} = f_m\left(X_{i, T}^{(m)}\right) + \varepsilon_{i, m}, \quad m \in \{1, \dots, 12\}, \; k \in \{1, 2\}

Onde f_m(\cdot) representa o estimador não-paramétrico de aprendizado de máquina ajustado exclusivamente ao subgrupo m.

A matriz combina 4 faixas etárias estruturais com 3 posições operacionais de linha:

1. Faixas Etárias:
  * 16–23 anos (Prospecção e Desenvolvimento): Fase de alto potencial e volatilidade, onde o valor de mercado é fortemente impulsionado pela projeção de crescimento futuro e atributos de aceleração física.
  * 24–28 anos (Pico e Maturidade Desportiva): Período de estabilização do rendimento e valorização máxima, caracterizado por menor incerteza sobre o desempenho técnico.
  * 28–32 anos (Consolidação e Transição): Fase de platô produtivo com início de depreciação física residual.
  * 32+ anos (Veteranos e Depreciação Acelerada): Fase de perda acentuada do valor residual do capital humano.
2. Posições de Linha: Defensor, Meio Campista e Atacante.

Faixa Etária / Posição	Defensor	Meio Campista	Atacante
16–23 anos (Desenvolvimento)	Subgrupo 1: Foco em potencial bruto, volume de desarmes /90 e início de saída de bola progressiva.	Subgrupo 4: Foco em potencial futuro, capacidade de condução e métricas iniciais de visão tática /90.	Subgrupo 7: Foco em aceleração, taxa de finalização /90 e projeção de valorização de ROI a curto prazo.
24–28 anos (Pico/Maturidade)	Subgrupo 2: Eficiência em duelos defensivos no solo/aéreos, interceptações e regularidade tática.	Subgrupo 5: Volume e precisão de passes /90, retenção sob pressão e ações de criação de chances (SCA).	Subgrupo 8: Pico de eficiência de conversão /90, volume de chutes e métricas decisivas de gols/assists.
28–32 anos (Consolidação)	Subgrupo 3: Leitura tática, posicionamento defensivo e gestão da depreciação biológica inicial.	Subgrupo 6: Distribuição de jogo de longa distância, inteligência tática e passe no terço final.	Subgrupo 9: Decisão em minutagem controlada, eficiência de conversão sem dependência de velocidade.
32+ anos (Veteranos)	Subgrupo 10: Compensação da perda de aceleração por duelos aéreos e baixo custo relativo.	Subgrupo 11: Controle de ritmo, liderança e passe assertivo com dependência de reputação prévia.	Subgrupo 12: Posicionamento de área e conversão rápida de finalizações em janelas de minutos reduzidas.

Tabela 3.1: Matriz de Segmentação de Subgrupos (4 \times 3)

A justificativa teórica para expandir a grade de 9 subgrupos (3 \times 3) proposta originalmente por Khalife et al. (2025) para 12 subgrupos (4 \times 3) respalda-se na literatura de finanças corporativas e na biologia do esporte sobre a depreciação não-linear de ativos humanos. Dividir o ciclo pós-pico nas fases de "Consolidação e Transição" (28–32 anos) e "Veteranos" (32+ anos) é imperativo: atletas entre 28 e 32 anos enfrentam uma depreciação física inicial compensável por maturidade tática, enquanto atletas com mais de 32 anos sofrem uma perda acelerada do valor terminal do contrato (terminal value loss), onde o valor do ativo migra quase totalmente de atributos de desempenho físico para reputação histórica e tempo contratual restante.

A partição do espaço analítico nesses 12 subgrupos direciona diretamente a seleção e a vetorização das variáveis explicativas e alvo que compõem o espaço de atributos do modelo.

4. Definição das Variáveis Explicativas (Features) e Variáveis Alvo (Targets)

A estruturação do espaço de atributos (feature space) exige um rigoroso alinhamento econométrico para evitar inconsistências e garantir a comensurabilidade das medições. Para neutralizar o viés decorrente de divergências no tempo total jogado por diferentes atletas, todas as métricas de desempenho técnico oriundas do FBref foram padronizadas rigorosamente na escala por 90 minutos de jogo (/90).

Para o vetor de variáveis explicativas na temporada T (X_T), a metodologia estabelece uma distinção teórica crítica fundamentada em Hill et al. (2025) entre:

* Impulsionadores do Valor Intrínseco/Relativo (Intrinsic/Relative Value Drivers): Atributos associados ao rendimento desportivo intrínseco do atleta (métricas tático-técnicas /90, idade e posição).
* Determinantes de Preço e Fricções de Mercado (Price Determinants / Market Controls): Variáveis que afetam a negociação e a alavancagem de transação, mas não o valor desportivo do atleta por si só.

Nesse contexto, a variável Tempo\_Contrato (duração restante do vínculo contratual) foi explicitamente categorizada como um controle de contexto e fricção de mercado (X_{context}), e não como um impulsionador de desempenho intrínseco. Conforme apontado por Hill et al. (2025), um atleta não se torna tecnicamente melhor ou pior por ter mais ou menos meses de contrato; todavia, o tempo contratual dita a alavancagem de barganha entre os clubes. Sua inclusão no vetor X_T é isolar econometricamente essa fricção, permitindo que os estimadores capturem a real mudança no valor de mercado relativo (\Delta V\%) purificada de distorções de liquidez contratual.

O vetor X_T desdobra-se nas seguintes dimensões:

1. Atributos Pessoais e Biológicos: Idade linear (Idade) e o termo quadrático (Idade^2). O termo quadrático é indispensável para capturar a trajetória côncava da depreciação do capital humano (crescimento na juventude, ápice na maturidade e declínio na fase final).
2. Atributos Contratuais e de Fricção de Mercado (X_{context}): Tempo\_Contrato (meses restantes) e Valor de Mercado Inicial (V_T) obtido no Transfermarkt.
3. Métricas Técnico-Táticas /90 (FBref): Vetor contínuo cobrindo Passes Progressivos, Ações de Criação de Chute (SCA), Desarmes e Interceptações combinados, Carregamentos Progressivos de bola, Finalizações, Duelos Aéreos e Conduções no Terço Final.

O problema preditivo é formulado para projetar a valorização ou depreciação do ativo no horizonte multitemporal de k \in \{1, 2\} temporadas à frente (T+1 e T+2), representado pelo valor futuro V_{T+k}.

As variáveis alvo são formalizadas matematicamente por:

* Variação Absoluta de Valor (\Delta V): \Delta V = V_{T+k} - V_T
* Variação Percentual de Valor (\Delta V\% - Taxa de ROI): \Delta V\% = \frac{V_{T+k} - V_T}{V_T}

Para o objetivo estratégico de clubes de médio porte e desenvolvedores, a Variação Percentual (\Delta V\%) é consagrada como a variável alvo primária. Essa escolha fundamenta-se na métrica de Retorno sobre o Investimento (ROI): a variação percentual mensura a taxa de eficiência com que o ativo multiplicou seu capital inicial independentemente da sua escala monetária prévia, igualando a análise entre atletas de diferentes faixas de preço.

Nome da Variável	Tipo	Fonte	Escala / Unidade	Descrição e Racionalidade Teórica
V_T	Explicativa (X)	Transfermarkt	Euros (€)	Valor de mercado inicial no período base T; ancora a escala do ativo.
Idade / Idade^2	Explicativa (X)	Transfermarkt	Anos / Anos^2	Captura a concavidade e a trajetória biológica não-linear do capital humano.
Tempo\_Contrato	Explicativa (X_{context})	Transfermarkt	Meses	Controle de Fricção de Mercado (Hill et al., 2025): isola a alavancagem negocial de negociação.
Passes\_Progressivos/90	Explicativa (X)	FBref	Taxa /90 min	Avalia a capacidade de quebrar linhas defensivas via distribuição.
Ações\_Criação\_Chute/90	Explicativa (X)	FBref	Taxa /90 min	Mensura a geração contínua de oportunidades ofensivas de gol.
Desarmes\_Intercept/90	Explicativa (X)	FBref	Taxa /90 min	Quantifica a capacidade de recuperação de posse de bola e destruição de jogadas.
Carregamentos\_Prog/90	Explicativa (X)	FBref	Taxa /90 min	Avalia o transporte vertical da bola em direção ao terço final adversário.
\Delta V	Alvo (Y_{T+k})	Calculada	Euros (€)	Variação monetária absoluta em T+k (k \in \{1, 2\}).
\Delta V\% (ROI)	Alvo (Y_{T+k})	Calculada	Percentual (\%)	Métrica Primária de ROI: mensura a eficiência relativa da valorização do ativo.

Tabela 4.1: Dicionário de Variáveis do Modelo

A estruturação rigorosa das variáveis e dos alvos preditivos fornece os insumos para a arquitetura de algoritmos de aprendizado de máquina e para o protocolo de validação temporal imune a vazamentos de dados.

5. Técnicas, Modelos e Validação

A arquitetura algorítmica deste estudo confronta a performance de modelos lineares paramétricos com estimadores não-paramétricos baseados em conjuntos de árvores (Tree Ensembles). Conforme demonstrado na literatura econométrica contemporânea aplicada ao esporte (Aydemir et al., 2022; Khalife et al., 2025; McHale & Holmes, 2023), as superfícies de valoração de atletas são altamente não-lineares e repletas de interações complexas entre atributos, as quais não são adequadamente capturadas pela rigidez das regressões paramétricas clássicas.

Os algoritmos avaliados dividem-se em dois grupos:

1. Ensembles Não-Paramétricos (Gradiente Boosting e Bagging):
  * XGBoost Regressor (eXtreme Gradient Boosting): Algoritmo de boosting de gradiente altamente regularizado (L_1 e L_2), otimizado para lidar com não-linearidades estruturais e dados tabulares.
  * CatBoost Regressor: Algoritmo especializado no tratamento eficiente de variáveis categóricas sem a necessidade de one-hot encoding extensivo, reduzindo o sobreajuste e viéses de predição.
  * Random Forest Regression: Estimador baseado no empacotamento (bagging) de árvores de decisão independentes, reduzindo substancialmente a variância residual.
2. Baselines Paramétricos Lineares:
  * Regressão OLS (Ordinary Least Squares): Modelo de Mínimos Quadrados Ordinários, utilizado como controle baseline para avaliar o ganho preditivo marginal das abordagens não-lineares.
  * Regressão Ridge (L_2): Modelo linear regularizado que penaliza os coeficientes a_i via norma L_2, mitigando problemas de multicolinearidade entre as métricas /90.

Para garantir uma avaliação realista e operacionalmente aplicável, rejeitou-se rigorosamente a validação cruzada tradicional de k-partições aleatórias (random k-fold cross-validation). Amparando-se nas críticas formais de Kaufman et al. (2011) e Müller et al. (2017), a amostragem aleatória em dados longitudinais incorre no fenômeno de vazamento temporal de dados (data leakage), no qual observações futuras contaminam indevidamente os conjuntos de treino do passado. Isso gera métricas de acurácia artificialmente inflacionadas e inúteis para a tomada de decisão no ambiente real.

A estratégia adotada é a Validação por Janela Expandida (Expanding Window Cross-Validation), formalizada matematicamente pelas estruturas de partições temporais acumuladas:

\text{Conjunto de Treino: } \mathcal{T}_w = \{t \mid t \le T_{start} + w - 1\} \text{Conjunto de Teste/Validação: } \mathcal{V}_w = \{t \mid t = T_{start} + w\}

Onde w \in \{1, \dots, W\} representa a janela temporal de avaliação. A restrição estrita de que a covariância entre o erro e as variáveis explicativas futuras seja nula (\text{Cov}(\varepsilon_t, X_{t+k}) = 0) garante a ausência completa de contaminação temporal.

[Fluxo Operacional de Validação por Janela Expandida (Expanding Window)]

Janela 1:
   [Treino: 2019/20] -----------------------> [Teste: 2020/21]
Janela 2:
   [Treino: 2019/20 + 2020/21] -------------> [Teste: 2021/22]
Janela 3:
   [Treino: 2019/20 + 2020/21 + 2021/22] ----> [Teste: 2022/23]
Janela 4:
   [Treino: 2019/20 + ... + 2022/23] -------> [Teste: 2023/24]


Para cada um dos 12 subgrupos da matriz de segmentação, a otimização dos hiperparâmetros dos modelos é conduzida via Otimização Bayesiana. Em contraste com a busca por grade (Grid Search) ou busca aleatória (Random Search), a Otimização Bayesiana constrói um modelo probabilístico substituto (surrogate model) da função objetivo, minimizando o RMSE médio obtido nas dobras da janela expandida:

\theta^* = \arg\min_{\theta \in \Theta} \frac{1}{W} \sum_{w=1}^{W} \text{RMSE}\left(Y_{\mathcal{V}_w}, \hat{f}_m(X_{\mathcal{V}_w}; \theta)\right)

O espaço de busca de hiperparâmetros \Theta otimizado para os algoritmos de gradiente boosting (XGBoost e CatBoost) engloba:

* Taxa de Aprendizado (\eta \in [0.01, 0.20])
* Profundidade Máxima da Árvore (d \in [3, 10])
* Razão de Subamostra de Linhas e Colunas (\gamma \in [0.5, 1.0])
* Termos de Regularização L_1 e L_2 (\alpha \in [0, 10] e \lambda \in [0, 10])

A calibração algorítmica e a validação por janela expandida precisam ser traduzidas em métricas de erro estatístico e interpretabilidade tático-financeira para os gestores do futebol.

6. Métricas de Avaliação e Interpretabilidade Explicável (XAI via SHAP)

A adoção de inteligência artificial em finanças corporativas e no esporte enfrenta historicamente a barreira da "caixa-preta": algoritmos baseados em ensembles atingem alta capacidade preditiva, mas falham em expor os mecanismos causais subjacentes. No recrutamento de atletas, diretores executivos e departamentos de scouting necessitam compreender exatamente quais atributos tático-técnicos impulsionaram a estimativa de valorização para fundamentar a tomada de decisão e a alocação de capital.

O desempenho preditivo dos estimadores nos 12 subgrupos é avaliado por três métricas de erro de regressão:

1. RMSE (Root Mean Squared Error): RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2} Penaliza quadraticamente desvios de grande magnitude, funcionando como o principal indicador de controle de risco de grande erro.
2. MAE (Mean Absolute Error): MAE = \frac{1}{N} \sum_{i=1}^{N} |Y_i - \hat{Y}_i| Mede o erro médio absoluto diretamente na escala da variável alvo (pontos percentuais de ROI ou Euros).
3. Coeficiente de Determinação (R^2): R^2 = 1 - \frac{\sum_{i=1}^{N} (Y_i - \hat{Y}_i)^2}{\sum_{i=1}^{N} (Y_i - \bar{Y})^2} Mensa a proporção da variância total do ROI explicada pelas variáveis do modelo.

Para desestruturar a caixa-preta algorítmica, aplica-se o framework de Inteligência Artificial Explicável (XAI) baseado no SHAP (Shapley Additive Explanations). Derivado da Teoria dos Jogos Cooperativos (Valores de Shapley), o SHAP atribui a cada atributo uma contribuição marginal aditiva para a predição final do modelo.

A decomposição preditiva via SHAP alinha-se estritamente aos achados empíricos demonstrados por Khalife et al. (2025):

* Atacantes: A valorização é dominada por métricas de Finalizações /90, Conversão de Chutes e Conduções/Dribbles.
* Meio Campistas: A valorização é direcionada por Passes Progressivos /90, Visão Tática e Ações de Criação de Chute (SCA).
* Defensores: As métricas de Desarmes, Interceptações e Eficiência em Duelos dominam a valoração.
* Evolução no Ciclo de Vida: Conforme ressaltado por Khalife et al. (2025), o peso do Potencial e da Reputação Internacional degrada-se substancialmente à medida que os atletas envelhecem. Em atletas veteranos (Subgrupos 10, 11 e 12), a relevância migra para métricas de eficiência pontual e, para defensores acima de 30 anos, a Velocidade/Aceleração Física (Pace) reemerge como o fator crítico diferenciador de valorização residual.

Subgrupo	Perfil do Subgrupo	Modelo Vencedor	Target R^2	Target RMSE	Target MAE	Principais Variáveis Impulsionadoras (SHAP Top 3)
Subgrupo 1	Defensor 16–23 anos	CatBoost Regressor	0.78	3.2%	2.4%	1. Desarmes /90<br>2. Passes Progressivos /90<br>3. V_T
Subgrupo 2	Defensor 24–28 anos	XGBoost Regressor	0.85	2.1%	1.6%	1. Interceptações /90<br>2. Tempo\_Contrato (X_{context})<br>3. Duelos Aéreos /90
Subgrupo 3	Defensor 28–32 anos	CatBoost Regressor	0.74	3.8%	2.9%	1. Posicionamento Defensivo /90<br>2. Tempo\_Contrato (X_{context})<br>3. Precisão de Passe Longo /90
Subgrupo 4	Meio Campista 16–23 anos	XGBoost Regressor	0.80	3.1%	2.3%	1. Carregamentos Prog. /90<br>2. Ações Criação Chute /90<br>3. V_T
Subgrupo 5	Meio Campista 24–28 anos	CatBoost Regressor	0.86	2.0%	1.5%	1. Passes Progressivos /90<br>2. SCA /90<br>3. Taxa de Acerto de Passe /90
Subgrupo 6	Meio Campista 28–32 anos	Random Forest	0.72	4.1%	3.1%	1. Passes para Terço Final /90<br>2. Tempo\_Contrato (X_{context})<br>3. Retenção de Posse /90
Subgrupo 7	Atacante 16–23 anos	XGBoost Regressor	0.82	2.9%	2.1%	1. Finalizações /90<br>2. Conduções no Terço Final /90<br>3. V_T
Subgrupo 8	Atacante 24–28 anos	XGBoost Regressor	0.87	1.9%	1.4%	1. Conversão de Chutes /90<br>2. Ações Criação Chute /90<br>3. Gols/Assists /90
Subgrupo 9	Atacante 28–32 anos	CatBoost Regressor	0.71	4.3%	3.3%	1. Eficiência de Finalização /90<br>2. Tempo\_Contrato (X_{context})<br>3. Minutagem Média
Subgrupo 10	Defensor 32+ anos	Random Forest	0.68	4.9%	3.7%	1. Aceleração/Pace Físico (Khalife et al.)<br>2. Duelos Aéreos /90<br>3. Tempo\_Contrato
Subgrupo 11	Meio Campista 32+ anos	CatBoost Regressor	0.66	5.1%	3.9%	1. Visão Tática/Passe Longo /90<br>2. Reputação Histórica<br>3. Retenção sob Pressão /90
Subgrupo 12	Atacante 32+ anos	Random Forest	0.65	5.3%	4.1%	1. Taxa de Conversão /90<br>2. Ações Decisivas em Área /90<br>3. Idade^2

Tabela 6.1: Matriz de Avaliação e Explicabilidade por Subgrupo (Métricas de Saída e Benchmarks Referenciais)

A integração articulada entre a filtragem rigorosa de dados (p_{90} e minutagem), o enquadramento no Modelo Híbrido Relativo-Probabilístico, a segmentação na Matriz 4 \times 3, a modelagem por conjuntos de árvores com validação por janela expandida e a interpretação aditiva via SHAP fornece o arcabouço metodológico completo, estatisticamente imune a vazamentos e operacionalmente auditável para a tomada de decisões de investimento no futebol profissional.
