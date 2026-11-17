# 02 — Verificação dos Dados e Análise Exploratória

**Notebook:** `notebooks/02_eda.ipynb` · **Código:** `src/football_analytics/sample.py`
**Saídas:** 14 figuras em `figures/02_eda/` e 19 tabelas em `tables/02_eda/`

[← Índice](README.md)

## 1. Objetivo

O notebook valida empiricamente as decisões amostrais da especificação: o corte $p_{90}$, os 90
minutos mínimos, a exclusão de goleiros e as 12 categorias. Também diagnostica a qualidade dos dados
e caracteriza as variáveis-alvo. As regras de amostra ficam em `sample.py` e são as mesmas usadas
pelo notebook 03.

**Teste pré-registrado (versão 3).** Os alvos de $T+1$ de 2024/25 (valor em 30/06/2026) formam o teste
final e ficam **ocultos** em todas as análises de alvo deste notebook (`sample.mask_holdout_targets`):
nenhuma decisão de modelagem pode depender deles. As verificações de amostra (corte, minutagem,
ausentes, categorias) incluem 2024/25, porque usam só informação de $T$. O alvo de $T+2$ de 2024/25
(30/06/2027) ainda não existe.

### Convenções adotadas

| Elemento | Definição |
|---|---|
| Unidade de análise | jogador–temporada. Transferidos entre clubes das Big Five têm os minutos somados e os atributos do clube com mais minutos |
| $V_T$, $V_{T+k}$ | último valor do Transfermarkt publicado até 30/jun do ano de término, com defasagem máxima de 183 dias |
| Idade | anos completos em 30/jun de $T$ (faixas); a *feature* contínua usa a idade decimal |
| Posição | posição granular do Transfermarkt → DEF/MEI/ATA (fallback: 1ª posição do FBref) |
| Faixas etárias | leitura formal dos intervalos da Tabela 6.2: 16–23, 24–28, 29–32 e 33+ anos completos |
| Corte $p_{90}$ | percentil 90 de $V_T$ **por temporada**, após as exclusões de goleiros e de minutagem |

## 2. Funil amostral

![Funil amostral](figures/02_eda/fig01_funil_amostral.png)

| Etapa | Observações | Atletas | Retenção |
|---|---|---|---|
| Registros FBref (jogador–clube–temporada) | 19.727 | 6.643 | — |
| Jogador–temporada | 18.808 | 6.643 | 95,3% |
| Pareados com o Transfermarkt | 18.763 | 6.609 | 99,8% |
| Exclusão de goleiros | 17.339 | 6.101 | 92,4% |
| Minutagem ≥ 90 min em $T$ | 15.286 | 5.138 | 88,2% |
| Com $V_T$ observado | 15.184 | 5.088 | 99,3% |
| **Corte $p_{90}$ de $V_T$** | **13.748** | **5.007** | 90,5% |

A amostra analítica tem **13.748 observações jogador–temporada de 5.007 atletas**, 40% a mais que a
versão 2 (9.810 e 4.084), com a inclusão de 2018/19 e 2024/25. O filtro que mais remove é o de
minutagem (−11,8%), seguido do corte $p_{90}$ (−9,5%) e da exclusão de goleiros (−7,6%). O pareamento
com o Transfermarkt quase não perde nada (99,8%).

## 3. O corte do percentil 90

![Distribuição de V_T](figures/02_eda/fig02_distribuicao_vt.png)

- **Forma.** $V_T$ é aproximadamente log-normal: a assimetria é 3,55 em nível e −0,05 em $\log_{10}$.
- **Limiar.** O $p_{90}$ efetivo é de **€ 27–32 milhões** por temporada (€ 30 mi agrupado), e não os
  € 18 mi de Müller et al. (2017). Na amostra, € 18 mi corresponde ao percentil 81 (19,2% das
  observações ficam acima).
- **Estabilidade.** As curvas acumuladas das sete temporadas são quase sobrepostas; o $p_{90}$ sobe de
  € 27–30 mi (2018/19 a 2022/23) para € 32 mi (2023/24 e 2024/25).

![p90 por liga e temporada](figures/02_eda/fig03_p90_liga_temporada.png)

A Premier League fica acima do limiar em todas as temporadas (€ 40–55 mi) e a Ligue 1 sempre abaixo
(€ 15–22 mi). **O corte não é neutro:**

![Perfil dos excluídos](figures/02_eda/fig04_perfil_excluidos_p90.png)

| Grupo | % acima do $p_{90}$ (excluído) |
|---|---|
| Premier League | 20,9 |
| La Liga / Bundesliga / Serie A | 7,3–8,1 |
| Ligue 1 | 3,7 |
| Atacantes / meio-campistas / defensores | 11,9 / 10,3 / 7,0 |
| 16–23 / 24–28 / 29–32 / 33+ anos | 11,5 / 12,4 / 4,7 / 0,9 |

| Cenário de corte | Limiar | Observações | % da Premier League na amostra |
|---|---|---|---|
| **$p_{90}$ por temporada (adotado)** | € 27–32 mi | 13.748 | 17,1% |
| $p_{90}$ único | € 30 mi | 13.776 | 17,1% |
| € 18 mi fixo | € 18 mi | 12.266 | 14,7% |
| $p_{90}$ por liga e temporada | € 15–55 mi | 13.815 | 19,6% |

Os excluídos se valorizam um pouco mais (27,9% contra 25,7% dos mantidos, nas temporadas de
desenvolvimento). O € 18 mi fica como referência histórica; o $p_{90}$ por temporada é a regra
operacional.

## 4. Minutagem mínima

![Minutagem](figures/02_eda/fig05_minutagem.png)

Entre os jogadores de linha com $V_T$, **10,5% têm menos de 90 minutos**. A remoção atinge sobretudo os
16–23 anos (22,7% da faixa, contra cerca de 5% nas demais) e, por posição, mais atacantes (14,4%) do que
defensores (6,6%). O piso de 90 min funciona como piso técnico; as taxas de quem joga entre 90 e 500
minutos continuam ruidosas e passam, na versão 3, por **encolhimento empírico-bayesiano** (relatório
03, §5.1).

**Implicação.** As conclusões sobre C1–C3 valem para "jovens com minutagem na elite", uma limitação de
generalização.

## 5. Valores ausentes

![Ausentes](figures/02_eda/fig06_ausentes.png)

- **Atributos demográficos e métricas centrais:** menos de 1% de ausência.
- **Ausência estrutural no Sofascore:** xG/xA só a partir de 2022/23, recuperações só a partir de
  2023/24 e bloqueios quase sempre ausentes. Essas variáveis foram **descartadas**: imputá-las criaria
  um sinal artificial de temporada. O xG/xA usado vem do Understat, que cobre todas as temporadas.
- **Contrato atual:** 27,8% de ausência, com gradiente de 49% em 2018/19 a 10% em 2024/25 (só o
  contrato vigente hoje é publicado). O notebook 03 reconstrói o contrato em $T$.
- **Alvos:** $V_{T+1}$ e $V_{T+2}$ faltam em 1,8% e 4,0% das observações das temporadas de
  desenvolvimento. As células de 2024/25 ficam vazias no mapa: $V_{T+1}$ é o teste pré-registrado e
  $V_{T+2}$ ainda não foi publicado.

**Atrito por categoria** (`tab09b`): a ausência do alvo se concentra nos veteranos.

| Categoria | Sem $V_{T+1}$ | Sem $V_{T+2}$ |
|---|---|---|
| C1–C9 (até 32 anos) | 0,3–2,0% | 0,3–5,6% |
| C10 · DEF 33+ | 10,4% | 26,3% |
| C11 · MEI 33+ | 12,4% | 24,9% |
| C12 · ATA 33+ | 10,3% | 22,2% |

A ausência é **informativa**: o veterano some porque se aposentou ou foi para uma liga sem avaliação.
Isso gera **viés de sobrevivência**, tratado no *backtest* do notebook 05 por intenção de tratar.

## 6. Matriz de 12 categorias

![Matriz 12 categorias](figures/02_eda/fig07_matriz_12_categorias.png)

Todas as células têm tamanho suficiente, com forte desbalanceamento: de **2.233 observações (C4 · DEF
24–28, 16,2%)** a **341 (C12 · ATA 33+, 2,5%)**. Os veteranos (C10–C12) somam 9,4% da amostra. Por
posição, há mais defensores (40,2%) que meio-campistas (31,5%) e atacantes (28,3%).

![Idade por posição](figures/02_eda/fig08_idade_por_posicao.png)

As idades medianas são 27 anos (DEF) e 26 anos (MEI e ATA). Os limites das faixas cortam regiões densas
da distribuição, e por isso a convenção de idade precisa constar no texto.

## 7. Variáveis-alvo

![Distribuição dos alvos](figures/02_eda/fig09_distribuicao_alvos.png)

Temporadas de desenvolvimento (2018/19 a 2023/24):

| | $\Delta V\%$ $T+1$ | $\Delta V\%$ $T+2$ | log-retorno $T+1$ | log-retorno $T+2$ |
|---|---|---|---|---|
| Mediana | −20,0% | −37,5% | −0,22 | −0,47 |
| Média | +5,8% | +14,5% | −0,17 | −0,40 |
| Assimetria | 19,8 | 20,9 | 1,10 | 0,72 |
| Curtose (excesso) | 662 | 639 | 4,1 | 1,7 |
| Variação nula | 9,3% | 4,7% | — | — |

$\Delta V\%$ é extremamente assimétrico, e média e mediana têm sinais opostos. Por isso o
**log-retorno** é o alvo de treino, e as previsões são reconvertidas para $\Delta V\%$.

![Direção da variação](figures/02_eda/fig10_direcao_variacao.png)

| Temporada $T$ | Desvalorizaram até $T+1$ | Valorizaram até $T+1$ |
|---|---|---|
| **2018/19** | **81,8%** | **15,8%** |
| 2019/20 | 58,9% | 36,1% |
| 2020/21 a 2023/24 | 62,1–63,1% | 24,9–26,7% |

**A temporada 2018/19 é a exceção nova da versão 3.** Seu $V_{T+1}$ é a revisão extraordinária de abril
de 2020, na pandemia: 82% dos atletas perderam valor. Em 2019/20, o movimento inverso (recuperação a
partir do valor deprimido). O notebook 04 testa um modelo sem 2018/19 no treino como robustez. Fora
dessas duas temporadas, a **taxa-base** é estável: cerca de 1 em cada 4 atletas se valoriza em um ano.

![Alvos por categoria](figures/02_eda/fig11_alvos_por_categoria.png)

- **A idade é o eixo dominante.** Nas faixas de 16–23 anos, a mediana de $\Delta V\%$ em $T+1$ é 0% e
  42–49% se valorizam; nas de 29 anos ou mais, a mediana fica entre −33% e −40% e no máximo 6,8% se
  valorizam.
- **A posição pesa pouco** dentro de cada faixa.
- **A dispersão é muito maior nos jovens:** em C2 (MEI 16–23), a média de $\Delta V\%$ em $T+1$ é +82% e a
  mediana, 0%.

![Curva de idade](figures/02_eda/fig12_curva_idade.png)

O valor mediano tem forma de U invertido, com pico entre 23 e 26 anos, e a variação mediana cai quase
continuamente com a idade. Isso justifica idade e idade² no modelo.

## 8. Associação entre desempenho e valorização

![Correlação desempenho × valorização](figures/02_eda/fig13_correlacao_desempenho_valorizacao.png)

- **Sem controles,** as métricas de desempenho praticamente não se associam a $\Delta V\%_{T+1}$
  (minutagem: $\rho$ de −0,02 a −0,03), enquanto a idade tem $\rho \approx -0{,}40$ a $-0{,}44$.
- **Controlando idade, idade², $\log V_T$ e temporada** (Spearman parcial), surge um sinal positivo
  consistente: nota do Sofascore 0,15–0,18; minutagem 0,14–0,16; xG/90 0,04–0,08.

A idade age como **confundidor**: os mais velhos jogam mais e se desvalorizam; os jovens jogam menos e
se valorizam. Com idade e preço controlados, o mercado premia sobretudo **titularidade** e
**desempenho agregado**.

## 9. Defasagem das avaliações

![Defasagem](figures/02_eda/fig14_defasagem_avaliacoes.png)

A mediana da defasagem é de 25–26 dias, e 75% das avaliações têm no máximo 30–33 dias. A exceção é o
pico em ~83 dias no painel $V_T$ (a revisão de 08/04/2020, base de 2019/20).

## 10. Conclusões para o artigo

1. Os filtros reproduzem as ordens de grandeza da especificação; o limiar real do $p_{90}$ é de
   € 27–32 mi.
2. O corte é assimétrico entre ligas (21% da Premier League); o corte por liga e temporada fica como
   sensibilidade.
3. O alvo de treino deve ser o log-retorno.
4. A idade é o eixo dominante da variação de valor, e o sinal de desempenho só aparece condicionado a
   ela.
5. Há viés de sobrevivência nos veteranos (até 26% sem $V_{T+2}$), dois choques de regime ligados à
   pandemia (2018/19 → 2019/20 e 2019/20 → 2020/21) e variáveis do Sofascore sem cobertura histórica,
   que foram descartadas.
