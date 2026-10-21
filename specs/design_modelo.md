6. Design do Modelo de Machine Learning

6.1. Arquitetura Geral da Modelagem e Definição das Variáveis-Alvo

A modelagem econométrica e preditiva desenvolvida neste capítulo adota o paradigma da regressão supervisionada para estimar e prever a valoração econômica de atletas de futebol profissional. Em termos teóricos, a arquitetura proposta insere-se no arcabouço das Tipologias de Valoração Relativa e Probabilística formuladas por Damodaran e sintetizadas no contexto do esporte por Hill et al. (2025). Uma vez que os atletas de futebol constituem ativos operacionais não-geradores diretos de fluxo de caixa isolado — dada a impossibilidade técnica e metodológica de isolar os fluxos financeiros de um único indivíduo no produto coletivo do clube sem incorrer em ruídos arbitrários —, a valoração probabilística baseada em aprendizado de máquina supervisionado estima o valor do ativo e sua apreciação de capital em função de métricas de desempenho relativas a seus pares de mercado.

Do ponto de vista formal, o problema econométrico é estruturado assumindo que o valor de mercado de um atleta i no instante futuro t+k (no qual k \in \{1, 2\} representa o horizonte temporal preditivo em temporadas) é governado por uma função não-linear contínua f(\mathbf{X}_{i,t}) aplicada ao vetor multidimensional de covariáveis do atleta \mathbf{X}_{i,t} no período base t, acrescida de um termo de erro estocástico \epsilon_{i,t}:

Y_{i,t+k} = f(\mathbf{X}_{i,t}) + \epsilon_{i,t}

O vetor de características \mathbf{X}_{i,t} engloba dimensões heterogêneas, incluindo atributos físicos, métricas táticas operacionais, visibilidade pública/popularidade e indicadores contratuais. Em consonância com a literatura econométrica contemporânea (Hill et al., 2025; Müller et al., 2017; Khalife et al., 2025), a modelagem não se limita a prever o valor nominal absoluto do atleta (V), mas foca precipuamente na dinâmica estocástica de variação temporal de seu valor de mercado.

Para capturar tanto a magnitude bruta do fluxo de valor nominal quanto o rendimento atuarial relativo sobre o patrimônio do clube detentor dos direitos econômicos, definem-se quatro variáveis-alvo (dependentes), divididas entre os horizontes de curto prazo (T+1) e médio prazo (T+2):

1. Variação Absoluta do Valor de Mercado no Horizonte T+1 (\Delta V_{i,t+1}): \Delta V_{i,t+1} = V_{i,t+1} - V_{i,t} Mede a alteração nominal bruta em euros (€) do valor de mercado do atleta i ao final da temporada subsequente (t+1) em relação ao valor observado na temporada base (t).
2. Variação Percentual do Valor de Mercado no Horizonte T+1 (\Delta V\%_{i,t+1}): \Delta V\%_{i,t+1} = \left( \frac{V_{i,t+1} - V_{i,t}}{V_{i,t}} \right) \times 100 Normaliza a variação financeira em termos relativos, permitindo comparar a taxa de valorização atuarial de curto prazo entre atletas pertencentes a diferentes patamares de preço.
3. Variação Absoluta do Valor de Mercado no Horizonte T+2 (\Delta V_{i,t+2}): \Delta V_{i,t+2} = V_{i,t+2} - V_{i,t} Mensura a trajetória nominal acumulada do valor do atleta ao longo de duas temporadas completas, capturando tendências estruturais de maturação ou depreciação econômica no médio prazo.
4. Variação Percentual do Valor de Mercado no Horizonte T+2 (\Delta V\%_{i,t+2}): \Delta V\%_{i,t+2} = \left( \frac{V_{i,t+2} - V_{i,t}}{V_{i,t}} \right) \times 100 Avalia a taxa de retorno percentual composta no horizonte de médio prazo (T+2), suavizando oscilações conjunturais e choques de desempenho pontuais de uma única temporada.

6.2. Estratégia de Filtragem e Recorte da Amostra

A construção da amostra de dados para treinamento e teste do modelo fundamenta-se estritamente na literatura empírica de finanças esportivas. A inclusão irrestrita de atletas no conjunto de dados introduz ruídos estruturais severos e vieses de precificação decorrentes da presença de superstars e de jogadores sem volume operacional mínimo.

A justificativa central para o corte amostral apoia-se no estudo empírico de Müller et al. (2017). Ao analisarem uma amostra de 845 transações reais de transferência nas cinco principais ligas europeias, os autores constataram que 769 transações ocorreram abaixo do percentil 90 (P_{90}), o qual corresponde ao limiar empírico de €18 milhões. Müller et al. (2017) demonstraram estatisticamente que modelos algorítmicos baseados em dados operacionais superam o julgamento humano e as estimativas de crowdsourcing (como o Transfermarkt) especificamente abaixo da marca do percentil 90. Acima desse limiar (€18 milhões), a precificação de superestrelas é fortemente distorcida por ativos intangíveis não-observáveis, tais como capacidade de atração de receitas comerciais globais, alcance em mídias sociais, poder de atração de patrocínios e prêmios irracionais de negociação decorrentes da utilidade do clube comprador. Por outro lado, para os 90% dos atletas de menor valor, o mercado comporta-se de forma racional e sistemática, tornando os modelos de aprendizado de máquina significativamente mais precisos do que as estimativas humanas.

A Tabela 6.1 detalha os critérios de amostragem, suas justificativas metodológicas e o embasamento na literatura.

Tabela 6.1: Critérios de Amostragem, Filtragem de Dados e Embasamento Teórico

Critério de Amostragem	Justificativa Metodológica e Fonte
Exclusão dos 10% Atletas Mais Caros (Limiar do Percentil 90 / €18 Milhões)	Restrição da amostra aos 90% atletas de menor valor. Com base em Müller et al. (2017) — em amostra de 845 transferências, onde 769 observações situaram-se abaixo de €18M —, modelos algorítmicos superam o consenso humano abaixo de P_{90}. Acima de €18M, prêmios de superstars, atração comercial e intangíveis de marca distorcem a relação entre estatísticas operacionais e preço.
Tempo Mínimo de Jogo (Ao menos 90 minutos acumulados na temporada)	Inclusão restrita a atletas com no mínimo 90 minutos de atuação na temporada. Conforme Müller et al. (2017) e Brandes & Franck (2012), a regra elimina variância extrema em taxas operacionais por 90 minutos (per 90) causada por amostras ínfimas de tempo em campo.
Exclusão da Posição de Goleiro	Remoção completa dos goleiros do conjunto de dados. Justifica-se pela incompatibilidade das métricas táticas operacionais de goleiros em relação aos atletas de linha, inviabilizando o compartilhamento do vetor de covariáveis \mathbf{X}_{i,t} (Bryson et al., 2012; Lucifora & Simmons, 2003; Müller et al., 2017).

6.3. Estratégia de Validação Cruzada Expandida e Estratificada (12 Categorias)

A validação de modelos preditivos aplicados a séries temporais esportivas exige um rigor metodológico estrito para evitar a contaminação temporal (data leakage), caracterizada pela utilização indevida de dados do futuro para prever eventos do passado (Müller et al., 2017; Kaufman et al., 2011). Simultaneamente, a dinâmica de mercado exige a mitigação de vieses categóricos decorrentes da idade e da posição tática dos atletas (Khalife et al., 2025).

Para solucionar a contaminação temporal, adota-se o esquema de Validação Cruzada por Janela Expandida (Expanding Window Cross-Validation). A ordem cronológica é preservada: em cada dobra k, o conjunto de treinamento acumula todas as temporadas anteriores disponíveis (T_1, \dots, T_k).

Correção Técnica para Horizontes de Médio Prazo (T+2): Na avaliação dos modelos que preveem alvos de curto prazo (T+1), o conjunto de teste é posicionado imediatamente na temporada T_{k+1}. Contudo, para os modelos que preveem a variação de valor em médio prazo (T+2), a utilização direta da temporada T_{k+1} como teste introduziria sobreposição temporal, pois a observação do alvo no instante T+2 requer dados acumulados do período T_{k+1}. Para sanar esse vazamento, implementa-se um buffer de expurgo temporal de 1 temporada (1-season temporal purging buffer). Assim, ao prever alvos T+2, a temporada T_{k+1} é completamente purgada do conjunto de teste, sendo o modelo treinado em [T_1, \dots, T_k] e avaliado estritamente na temporada T_{k+2}.

Paralelamente, para garantir que as dobras mantenham a distribuição de subgrupos do mercado, o processo é estratificado combinando 4 Faixas Etárias com 3 Posições de Linha, resultando em 12 categorias categóricas independentes. Formalmente, as faixas etárias são definidas em termos de intervalos matemáticos fechados/abertos na data de corte da temporada:

* Faixa Etária 1 (Jovem / Formação): I_1 = [16, 23] anos.
* Faixa Etária 2 (Ápice Físico e Tático): I_2 = (23, 28] anos.
* Faixa Etária 3 (Maturidade / Veterano): I_3 = (28, 32] anos.
* Faixa Etária 4 (Fase Final de Carreira): I_4 = (32, \infty) anos.

A Tabela 6.2 ilustra a matriz de estratificação 4 \times 3 geradora das 12 categorias categóricas.

Tabela 6.2: Matriz de Estratificação em 12 Categorias Combinadas por Idade e Posição

Faixa Etária / Estágio de Carreira	Defensor (DEF)	Meio-Campista (MEI)	Atacante (ATA)
[16, 23] anos (Jovem / Formação)	Categoria 1: DEF_16-23	Categoria 2: MEI_16-23	Categoria 3: ATA_16-23
(23, 28] anos (Ápice Físico/Tático)	Categoria 4: DEF_24-28	Categoria 5: MEI_24-28	Categoria 6: ATA_24-28
(28, 32] anos (Maturidade / Veterano)	Categoria 7: DEF_28-32	Categoria 8: MEI_28-32	Categoria 9: ATA_28-32
(32, \infty) anos (Fase Final de Carreira)	Categoria 10: DEF_32+	Categoria 11: MEI_32+	Categoria 12: ATA_32+

6.4. Fundamentação Teórica e Operacional da Segmentação Subgrupal

A justificativa financeira e econométrica para treinar e validar modelos segmentados nas 12 categorias apoia-se no arcabouço da maximização de utilidade no futebol europeu (Kesenne, 2000; Berri et al., 2023; Sloane, 1971). Na gestão do elenco, os clubes operam em um ambiente onde a maximização de vitórias se sobrepõe ao lucro no curto prazo, exigindo um equilíbrio de portfólio entre ativos de capital especulativos de alta volatilidade e upside (atacantes jovens) e ativos operacionais de baixo upside e rendimento constante (defensores veteranos).

Estudos empíricos recentes demonstram por que a segmentação subgrupal é indispensável. Khalife et al. (2025) comprovaram que a inclusão do "potencial futuro projetado" eleva a precisão preditiva (R^2) de modelos de aprendizado de máquina de uma média de 74% (em modelos sem potencial) para 84% (em modelos com potencial), reduzindo substancialmente o erro quadrático médio normalizado. Mais importante, Khalife et al. (2025) demonstraram que a estrutura de importância dos recursos altera-se drasticamente ao longo da carreira do atleta: enquanto a precificação de atletas jovens [16, 23] é amplamente dominada pela reputação internacional e pelo potencial intangível de crescimento, a partir dos 29 anos a importância do potencial decai a zero. Para atletas veteranos (>29 anos), a valoração é governada exclusivamente pela produção operacional tangível em campo — especificamente eficiência de finalização para atacantes, métricas de distribuição e passe para meio-campistas, e volume de ações defensivas e ritmo físico (pace) para defensores.

Os blocos a seguir sintetizam as premissas teóricas que sustentam a segmentação subgrupal.

Premissa Teórica 1: Precificação de Atacantes Jovens e a Dominância do Potencial Futuro Conforme demonstrado por Khalife et al. (2025) e Metelski (2021), a valoração de atacantes jovens na faixa [16, 23] anos é moldada pela expectativa de valor residual futuro e pelo potencial de revenda. O modelo preditivo nesses subgrupos atribui peso desproporcional ao potencial projetado e à reputação, superando a produção operacional de curto prazo. A incorporação do potencial eleva a acurácia global do modelo (R^2) de 74% para 84% (Khalife et al., 2025).

Premissa Teórica 2: Transição de Atributos e Valoração Operacional em Veteranos e Defensores À medida que os atletas superam os 28 anos, o fator "potencial futuro" perde poder explicativo. Conforme evidenciado por Khalife et al. (2025), a precificação de defensores e veteranos em (28, 32] e (32, \infty) reage estritamente à consistência operacional diária: taxa de acerto de passes, ações defensivas bem-sucedidas, duelos aéreos ganhos e durabilidade física (pace). Modelos não-segmentados falham ao tentar impor coeficientes médios globais a dinâmicas financeiras tão assimétricas.

6.5. Algoritmos, Baseline de Referência e Métricas de Erro

Para capturar a estrutura não-linear e a assimetria à direita (right-skewness) características dos dados do mercado de transferências (Hill et al., 2025; Müller et al., 2017; Aydemir et al., 2022), selecionou-se uma suíte de algoritmos contrastando baselines lineares e modelos não-paramétricos baseados em árvores de decisão.

A especificação funcional e o papel metodológico de cada algoritmo são detalhados a seguir:

* Modelos de Baseline Lineares:
  * Mínimos Quadrados Ordinários (OLS): Atua como referência estatística paramétrica tradicional, medindo relações lineares diretas sem penalização de coeficientes.
  * Regressão Ridge: Aplica uma penalização L_2 à norma dos coeficientes, mitigando problemas de multicolinearidade severa entre métricas táticas altamente correlacionadas no vetor de covariáveis \mathbf{X}_{i,t} (como volume de passes e passes progressivos).
* Ensembles Não-Paramétricos de Árvores de Decisão:
  * Random Forest: Algoritmo de bagging que reduz a variância do modelo construindo múltiplas árvores de decisão descorrelacionadas via amostragem bootstrap de dados e recursos.
  * XGBoost (eXtreme Gradient Boosting): Algoritmo de gradient boosting altamente eficiente que constrói árvores de forma sequencial, otimizando funções de perda diferenciáveis e capturando interações de alta ordem sem exigir normalidade dos dados (Khalife et al., 2025; McHale & Holmes, 2023).
  * CatBoost: Algoritmo de gradient boosting projetado para tratar nativamente variáveis categóricas e prevenir o vazamento do alvo (target leakage) por meio do cálculo de estatísticas de alvo baseadas no tempo histórico.

Para avaliar rigorosamente o desempenho preditivo nas dobras de validação, utilizam-se três métricas estatísticas padronizadas:

1. Erro Quadrático Médio (Root Mean Squared Error - RMSE): Penaliza desvios de grande magnitude com maior severidade devido à elevação quadrática dos erros. RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( Y_i - \hat{Y}_i \right)^2}
2. Erro Absoluto Médio (Mean Absolute Error - MAE): Mede a magnitude média absoluta dos erros de previsão, fornecendo uma métrica de desvio financeiro direto em euros (€) ou em pontos percentuais. MAE = \frac{1}{N} \sum_{i=1}^{N} \left| Y_i - \hat{Y}_i \right|
3. Coeficiente de Determinação (R^2): Mede a proporção da variância total da variável-alvo explicada pelas covariáveis do modelo. R^2 = 1 - \frac{\sum_{i=1}^{N} \left( Y_i - \hat{Y}_i \right)^2}{\sum_{i=1}^{N} \left( Y_i - \bar{Y} \right)^2} Em que \bar{Y} = \frac{1}{N} \sum_{i=1}^{N} Y_i representa explicitamente a média amostral das variações reais observadas na variável-alvo dentro da dobra de avaliação correspondente.

6.6. Otimização de Hiperparâmetros e Pipeline de Treinamento

Para prevenir o sobreajuste (overfitting) e garantir a convergência dos algoritmos não-paramétricos, o espaço de hiperparâmetros é explorado via Otimização Bayesiana. A Otimização Bayesiana utiliza um modelo probabilístico substituto (surrogate model via Processos Gaussianos) para mapear a função objetivo de erro e selecionar iterativamente os conjuntos de hiperparâmetros com maior probabilidade de melhoria.

O pipeline operacional sequencial de treinamento é estruturado em cinco etapas integradas:

1. Ingestão de Dados e Filtragem Inicial: Aplicação estrita das regras de amostragem (remoção dos 10% atletas mais caros acima do limiar de €18 milhões / P_{90}, exclusão de goleiros e aplicação do filtro de tempo mínimo de jogo de 90 minutos por temporada).
2. Particionamento Temporal via Janela Expandida Purgada: Divisão cronológica das dobras com aplicação do buffer de expurgo de 1 temporada para alvos T+2, eliminando a contaminação do conjunto de teste.
3. Pré-Processamento e Imputação Interna à Dobra: Imputação de dados ausentes utilizando a mediana estratificada por posição, calculada estritamente com os dados do conjunto de treinamento da dobra correspondente para evitar contaminação temporal.
4. Escalamento e Normalização Estratificada: Aplicação do StandardScaler ajustado individualmente para cada uma das 12 categorias dentro do conjunto de treino.
5. Otimização Bayesiana de Hiperparâmetros e Treinamento Final: Busca automatizada da combinação ideal de hiperparâmetros minimizando o RMSE interno de validação. O espaço de busca para os algoritmos não-paramétricos é delimitado conforme os seguintes intervalos operacionais:
  * XGBoost: max_depth \in [3, 10], learning_rate (\eta) \in [0.01, 0.30], n_estimators \in [100, 1000], subsample \in [0.6, 1.0], colsample_bytree \in [0.6, 1.0] e gamma \in [0, 5].
  * CatBoost: depth \in [4, 10], learning_rate \in [0.01, 0.30], l2_leaf_reg \in [1, 10] e iterations \in [200, 1200].
  * Random Forest: n_estimators \in [100, 800], max_depth \in [5, 25], min_samples_split \in [2, 10], min_samples_leaf \in [1, 5] e max_features \in [\text{'sqrt'}, \text{'log2'}, 0.3-0.8].

6.7. Interpretabilidade do Modelo via SHAP (Shapley Additive Explanations)

Embora os modelos não-paramétricos baseados em gradient boosting ofereçam elevada acurácia preditiva, sua estrutura em árvore é frequentemente tratada como uma "caixa-preta". Para prover transparência econômica e tática, adota-se a metodologia SHAP (Shapley Additive Explanations), fundamentada na teoria dos jogos cooperativos (Lundberg & Lee, 2017).

Matematicamente, a contribuição marginal de uma variável i no vetor de características F, denotada pelo valor Shapley \phi_i(x), é calculada considerando todas as combinações possíveis de subconjuntos de recursos S \subseteq F \setminus \{i\}:

\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]

Na qual f_x(S) representa a previsão do modelo treinada no subconjunto de recursos S. A previsão final do modelo para um indivíduo x, f(x), é decomposta de forma aditiva a partir da expectativa base \mathbb{E}[f(x)]:

f(x) = \mathbb{E}[f(x)] + \sum_{j=1}^{M} \phi_j(x)

A análise SHAP será conduzida de forma individualizada para cada uma das 12 categorias categóricas. Essa abordagem permite mapear com precisão cirúrgica quais métricas táticas do FBref/Wyscout são determinantes para a valoração em cada setor e estágio de carreira, distinguindo métricas de volume operacional (per 90 minutes) de métricas de eficiência percentual.

O Quadro 6.1 sumariza a matriz de domínio teórico esperado das variáveis táticas divididas entre métricas por 90 minutos e eficiência percentual.

Quadro 6.1: Matriz de Domínio Teórico Esperado das Variáveis Táticas por Posição (Taxas por 90 Minutos vs. Eficiência Percentual)

Posição Funcional	Métricas Táticas por 90 Minutos (Rate Metrics / Per 90)	Métricas de Eficiência Percentual (Efficiency % Metrics)	Impacto Teórico Esperado na Valoração
Defensores (DEF)	Desarmes por 90 min, Interceptações por 90 min, Cortes por 90 min, Duelos Aéreos Disputados por 90 min.	% de Duelos Aéreos Ganhos, % de Desarmes Bem-Sucedidos, % de Acerto de Passes Totais.	A eficiência percentual defensiva e a taxa de acerto de passe curto/longo reduzem o risco atuarial e adicionam prêmio de valor em defensores de fases (23, 28] e (28, 32] anos (Khalife et al., 2025).
Meio-Campistas (MEI)	Passes Progressivos por 90 min, Conduções Progressivas por 90 min, Passes Chave por 90 min, Ações de Criação de Chute (SCA) por 90 min.	% de Acerto de Passes Curtos, Médios e Longos, % de Dribles Certos.	O volume de distribuição progressiva e a retenção percentual da posse de bola sob pressão dominam a valoração financeira do setor central (Behravan & Razavi, 2021).
Atacantes (ATA)	Finalizações no Alvo por 90 min, Gols Esperados (xG) por 90 min, Gols Anotados por 90 min, Toques na Área Adversária por 90 min.	% de Conversão de Chutes em Gol, % de Dribles Bem-Sucedidos, % de Acerto em Duelos Ofensivos.	Métricas de volume e eficiência de conversão ofensiva constituem os principais vetores de precificação do setor de ataque, apresentando volatilidade máxima em jovens de [16, 23] anos (Khalife et al., 2025; Majewski, 2016).

6.8. Síntese do Design e Próximos Passos

A arquitetura metodológica desenhada neste capítulo estabelece um arcabouço rigoroso, replicável e empiricamente auditável para a modelagem preditiva da valoração econômica de atletas de futebol. O design proposto apoia-se em quatro pilares fundamentais:

* Recorte Amostral no Percentil 90 (€18 Milhões): Filtragem fundamentada em Müller et al. (2017), eliminando as distorções irracionais de superstars e concentrando o poder preditivo do aprendizado de máquina onde as estatísticas operacionais possuem eficácia comprovada.
* Validação Cruzada Expansiva com Purga e Estratificação em 12 Categorias: Preservação estrita da ordem cronológica temporal com aplicação de buffer de expurgo de 1 temporada para alvos T+2, garantindo ausência de vazamento de dados (data leakage) e capturando as assimetrias táticas de 4 faixas etárias e 3 posições.
* Suíte Algorítmica Não-Paramétrica Otimizada via Bayes: Utilização de algoritmos de gradient boosting (XGBoost, CatBoost) e bagging (Random Forest) com otimização automatizada de hiperparâmetros para capturar relações não-lineares nos dados.
* Interpretabilidade Granular por Subgrupo via SHAP: Abertura da "caixa-preta" algorítmica por meio de formulações rigorosas da Teoria dos Jogos Cooperativos, permitindo identificar os vetores táticos determinantes do valor econômico em cada estágio da carreira do atleta.

Com a arquitetura do modelo e o pipeline operacional devidamente formalizados, o estudo avança para a etapa de execução experimental, apresentação dos resultados preditivos obtidos e discussão detalhada dos achados empíricos.
