# CLAUDE.md - Football Transfer Risk & AI Scouting

## 1. Project Overview & Business Core
* **Theme:** Previsão da variação do valor de mercado futuro ($\Delta V$ e $\Delta V\%$) de jogadores de linha das cinco principais ligas europeias (*Big Five*) utilizando Machine Learning Supervisionado e eXplainable AI (XAI - SHAP).
* **Core Business Model:** *Buy-Low / Sell-High* (Estratégia de *Player Trading* voltada para clubes médios e desenvolvedores focada na identificação de ativos subavaliados com alto potencial de valorização).
* **Temporal Horizons:** Previsão da variação de valor nos horizontes $T+1$ (1 temporada) e $T+2$ (2 temporadas) com base em atributos e métricas consolidadas no momento de referência $T$.
* **Scope & Sample Filters:**
  * **Ligas:** Premier League, La Liga, Serie A, Bundesliga e Ligue 1 (Temporadas 2019/20 a 2023/24).
  * **Filtro de Custo ($p90$):** Restrição estrita aos **90% atletas de menor valor de mercado** em $T$ (exclusão dos 10% mais caros / superestrelas para eliminar ruídos de *branding* e cláusulas atípicas).
  * **Filtro de Minutagem:** Mínimo de **90 minutos jogados** na temporada $T$ no FBref.
  * **Exclusão Técnica:** Goleiros excluídos (análise focada exclusivamente em jogadores de linha).
* **Matriz de Segmentação (12 Categorias):**
  * **4 Faixas Etárias:** 16–23 anos (Desenvolvimento/Prospecção), 24–28 anos (Pico/Maturidade), 28–32 anos (Consolidação/Transição), 32+ anos (Veteranos/Depreciação).
  * **3 Posições de Linha:** Defensor, Meio Campista, Atacante.
* **Referências specs**: ./specs

---

## 2. User Profile & Technical Guidelines
* **User Profile:** Data Engineer / ML Engineer experiente realizando trabalho acadêmico avançado.
* **Code Standards:**
  * Código Python pronto para produção, modular, tipado (`typing`) e aderente à PEP 8.
  * Pipelines de Data Engineering eficientes e reutilizáveis.
  * Tratamento de exceções e logs estruturados em scripts de *web scraping* e processamento.
* **Tech Stack Principal:** `python 3.10+`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `catboost`, `shap`, `rapidfuzz`, `soccerdata` / `requests` / `beautifulsoup4`, `matplotlib`, `seaborn`.

---

## 3. Jupyter Notebooks Architecture (5-Step Pipeline)

### `01_data_extraction.ipynb`
* **Objetivo:** Coleta, extração e fusão inicial das fontes de dados.
* **Ações:**
  1. *Scraping* / Extração de métricas tático-técnicas do **FBref** (relatórios de passes, ações defensivas, criação de chances, xG, etc.) para as *Big Five* (2019/20 a 2023/24).
  2. *Scraping* / Extração do **Transfermarkt** (histórico de valor de mercado, data de término de contrato, data de nascimento, posição).
  3. Resolução de entidades (*Entity Resolution*) unificando atletas via alinhamento difuso de strings (`rapidfuzz`).
* **Saída:** Dataset bruto consolidado (`data/raw/football_raw_dataset.csv` / `.parquet`).

### `02_eda.ipynb`
* **Objetivo:** Análise Exploratória de Dados (EDA) e validação metodológica da amostra.
* **Ações:**
  1. Validação do corte estatístico do percentil 90 ($p90$) no valor inicial em $T$.
  2. Verificação do filtro de minutagem ($\ge 90$ minutos) e distribuição de ausentes (*missing values*).
  3. Análise da distribuição amostral nas **12 categorias** (4 idades $\times$ 3 posições).
  4. Análise de assimetria (*skewness*) e distribuição das variações de valor no mercado.
* **Saída:** Relatórios visuais, gráficos de distribuição e diagnóstico de qualidade de dados.

### `03_preprocessing.ipynb`
* **Objetivo:** Limpeza, engenharia de *features* e estruturação das janelas temporais.
* **Ações:**
  1. Cálculo de *features* em $T$: curva quadrática de idade ($\text{Idade}^2$), tempo restante de contrato (em anos) e padronização por 90 minutos ($/\text{90}$).
  2. Construção das Variáveis Alvo em $T+1$ e $T+2$:
     * Variação Absoluta: $\Delta V = V_{T+k} - V_T$
     * Variação Percentual: $\Delta V\% = \frac{V_{T+k} - V_T}{V_T}$
  3. Criação dos *lags* e estruturação do *dataset* em formato painel temporal.
* **Saída:** Datasets processados para modelagem (`data/processed/train_test_lags.parquet`).

### `04_model_training.ipynb`
* **Objetivo:** Treinamento, otimização e comparação entre algoritmos preditivos.
* **Ações:**
  1. Implementação da Validação Cruzada por Janela Expandida (*Expanding Window Cross-Validation*) respeitando a cronologia das temporadas.
  2. Treinamento estratificado/segmentado considerando a matriz das **12 categorias**.
  3. Comparação de modelos: **XGBoost Regressor**, **CatBoost Regressor** e **Random Forest Regressor** contra *baselines* lineares (OLS / Ridge).
  4. Otimização Bayesiana de hiperparâmetros.
  5. Avaliação por métricas de erro: $RMSE$, $MAE$ e $R^2$ (globais e por subgrupo).
* **Saída:** Modelos treinados salvos, tabela comparativa de métricas e predições consolidadas.

### `05_model_interpretation.ipynb`
* **Objetivo:** Interpretação dos resultados via eXplainable AI (XAI) e avaliação de risco de transferência.
* **Ações:**
  1. Extração e cálculo dos valores SHAP (`shap.TreeExplainer`) para os melhores modelos.
  2. Geração de *Summary Plots*, *Dependence Plots* e *Waterfall Plots* segmentados pelas **12 categorias**.
  3. Identificação das variáveis táticas do FBref mais determinantes na valorização futura.
  4. Simulação de cenários de recrutamento *Buy-Low / Sell-High* e quantificação de risco financeiro.
* **Saída:** Artefatos gráficos de XAI e diagnósticos gerenciais para tomada de decisão.

---

## 4. Academic Writing Rules (ABNT Standard)
* **Fonte:** Arial, Tamanho 12 para texto geral; Tamanho 14 Negrito para Títulos; Tamanho 10 para citações longas e notas de rodapé.
* **Espaçamento:** 1,5 entre linhas para o corpo do texto; 0 pts entre parágrafos; 1 linha em branco (enter) entre títulos e texto.
* **Citações Longas (+3 linhas):** Tamanho 10, espaçamento simples (1,0), recuo esquerdo exatamente de 4,0 cm, espaçamento de 6 pts antes e depois.
* **Margens:** Esquerda: 3,0 cm, Direita: 2,0 cm, Superior: 3,0 cm, Inferior: 2,0 cm.
* **Parágrafos:** Primeira linha com recuo de 1,0 cm.
* **Referências:** Alinhadas à esquerda, espaçamento simples (1,0) dentro da mesma referência, separadas por 1 linha em branco (6 pts), em ordem alfabética por sobrenome do autor.
* **Extensão do Artigo:** 15 a 20 páginas.
* **Tom Acadêmico:** Português formal, linguagem impessoal (3ª pessoa), precisão terminológica em finanças e inteligência esportiva.

---

## 5. Article Structure & Chapter Mapping
1. **Pré-texto:** Título, Autores, Resumo (100–250 palavras), Palavras-chave.
2. **Introdução:** Contextualização das *Big Five*, modelo *buy-low/sell-high*, lacuna (substituição do *nowcasting* pelo *forecasting* $T+1/T+2$), objetivos e relevância da XAI.
3. **Revisão da Literatura:** Métrica de desempenho no futebol, precificação via Transfermarkt / sabedoria das multidões, Machine Learning preditivo no esporte e XAI na tomada de decisão.
4. **Problema e Objetivos:** Conforme detalhado no relatório `3. Problema e Objetivos`.
5. **Diferencial do Estudo:** Conforme detalhado no relatório `Diferencial do Meu Estudo`.
6. **Verificação dos Dados:** Conforme detalhado no relatório `5. Verificação dos dados` ($p90$, $\ge 90$ min, exclusão de goleiros e 12 categorias).
7. **Metodologia e Design do Modelo:** Conforme detalhado nos relatórios `Definição da Metodologia` e `6. Design do Modelo de Machine Learning`.
8. **Resultados e Discussão:** Desempenho preditivo, decomposição SHAP por categoria e análise de risco para diretores de futebol/scouts.
9. **Conclusão e Referências.**


