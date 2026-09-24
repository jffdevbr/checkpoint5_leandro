# Plano — completar o pipeline (Checkpoint 5, entrega 24/09)

## Contexto

O notebook `notebooks/01_pipeline_completo.ipynb` (60 células) é um esqueleto com TODOs nas seções 3–10; a API (`app/main.py`) e as páginas rodam sobre modelos *baseline* (`scripts/bootstrap_models.py`). O objetivo é fechar o pipeline para responder às duas perguntas **isolando o efeito do que é decisão**:

- **P1:** quais variáveis **controláveis** reduzem `Energy_Intensity` mantendo a produção?
- **P2:** quais variáveis de **estado** indicam necessidade de manutenção, e a decisão pode ser automatizada?

Decisões já tomadas com você: **(a)** adicionar premissa de custo de feedstock; **(b)** separar a manutenção em **mecânica** (vibração) e **troca de catalisador** (idade).

---

## Achados já verificados nos dados (base de tudo que vem abaixo)

Rodei checagens read-only no CSV. O processo gerador está praticamente mapeado:

| Relação | Evidência | Status no notebook atual |
|---|---|---|
| `Yield = 0,18·Flow·Health` | resíduo máx. 3e-14 | já documentado |
| `EI = (3,6·Elec + 0,035·Gas)/Yield` | R² = 1 | já documentado |
| **`Health ~ U(0,70–1,00)` se vib ≤ 6,5; `U(0,56–0,80)` se vib > 6,5** | limiar exato, igual nas 3 plantas; média 0,851 → 0,682 (−20% produção); 24,7% das leituras estão degradadas | **novo** — a correlação −0,49 esconde um degrau |
| **`Gas = U(3000–5000) + 0,5·Catalyst_Age`** | resíduo exatamente em [3000,5; 4999,8] | **novo** — o "corr 0,087" era um efeito causal real |
| `Elec`, `Steam` ~ uniformes, independentes de tudo | KS p≈0,96; MI≈0 | já documentado (Steam fora da fórmula) |
| T, P, Válvula: efeito nulo | corr < 0,02 global e por planta | já documentado |
| `Unit_Name`, `Catalyst_Type`: sem efeito nos níveis | ANOVA p = 0,35 / 0,34 / 0,54 | parcial |
| Inclinação do gás por catalisador: 0,46 / 0,61 / 0,36 | ~1,9 erro-padrão de diferença → **testar** | novo (hipótese) |
| **`Ambient_Temp` sem sazonalidade** (média ~21 °C em todo mês/hora) | — | **contradiz** doc 1.2 ("mês explica ambient") |
| Teto de R² honesto para EI = **0,713** (oráculo `E[E|idade]/(0,18·F·H)`) | Ridge atual 0,688; **Ridge log-log (logF, logH, idade) 0,711**; HGB 0,695 | novo |
| KFold ≈ TimeSeriesSplit em todos os modelos | sem drift | resolve a contradição doc 1.2 × seção 5 |
| **Regime degradado (H≈0,68) torna Y ≥ 80 t inviável**: exige F ≥ 654 > limite 648,8 | — | **novo** — liga P1 a P2 |

**Narrativa resultante:** a única alavanca operacional é a vazão (elasticidade de EI em relação à vazão = −1); a manutenção mecânica recupera ~20% de produção; a troca de catalisador reduz o gás. T/P/Válvula não são alavancas neste dataset (limitação do dado sintético, dita explicitamente).

---

## Etapa 0 — Código compartilhado (evitar 3 cópias divergentes)

Hoje `build_features`/`montar_X` estão duplicados no notebook, em `app/main.py` e em `scripts/bootstrap_models.py`, e `prob_falha` está no notebook e na API. O `config.py` cita um `src/features.py::limites_operacionais` que não existe.

- **Criar `src/features.py`**: `build_features(df)`, `montar_X(x, ctx, features)`, `limites_operacionais(df, cols)` e as listas `FEATURES_ENERGIA` / `FEATURES_YIELD` / `FEATURES_ESTADO`.
- **Criar `src/decisao.py`**: `economia()`, `prob_falha()`, `projetar_estado()`, `avaliar_cenario()`, `idade_otima_catalisador()`, `gate_automacao()`, `otimizar_lp()`.
- O notebook, `app/main.py` e `bootstrap_models.py` passam a importar daí.

## Etapa 1 — EDA: hipóteses a incluir (seção 2 do notebook)

Cada uma vira uma célula curta: **hipótese → teste → conclusão impressa**, que depois é exportada para a página web.

- **H1 — Limiar de vibração.** Varrer o limiar de 6,0 a 7,0 e mostrar o degrau do Health; gráfico em faixas com min/max. Confirma o limiar de 6,5 para as regras de negócio.
- **H2 — Catalisador envelhece → mais gás.** Regressão `Gas ~ idade`, gráfico de faixas, resíduo `Gas − 0,5·idade` uniforme.
- **H3 — Interação `Catalyst_Type × idade` no gás** (OLS do statsmodels com interação, teste F).
  - Se for significativa, o tipo de catalisador vira decisão de longo prazo na troca.
  - Se não for, fica documentado que não há efeito.
- **H4 — Sazonalidade do ambiente.** Média por mês/hora → refutada, e as features temporais são descartadas com justificativa.
- **H5 — Setpoints sem efeito, inclusive não linear.** Informação mútua e dependência parcial do HGB para T, P e V, por planta.
- **H6 — Planta/catalisador como populações distintas.** ANOVA → não são; um modelo único serve às 3 plantas.
- **H7 — Independência temporal.** Autocorrelação lag-1 já existe no doc; somar KFold × TimeSeriesSplit mostrando que dão o mesmo resultado (sem drift).
- **H8 — Teto teórico de R²** para EI (0,713) e para Yield sem Health (~0,66). Passa a ser a régua de "bom o suficiente" da seção 5.
- **Limpeza:** a seção 2.3 atual (regressão linear de EI por vibração/idade) fica substituída por H1/H2, que mostram o **mecanismo** em vez de uma inclinação média.

## Etapa 2 — Feature engineering (seções 3 e 4.3)

Cada feature precisa ter **papel causal**, **não ser redundante** e **não vazar o target**. Célula nova: tabela `feature | modelo | tipo (decisão/estado/externa) | justificativa | corr máx. com outra feature / VIF | risco de leakage`.

**Modelo de Energia (P1)** — target `log(EI)`:

| Feature | Por quê |
|---|---|
| `log_Flow` | decisão; o coeficiente é a elasticidade direta (≈ −1) |
| `log_Health` | estado; confundidor que precisa ser controlado para isolar o efeito da vazão |
| `Catalyst_Age_Days` | estado; canal causal via gás |
| (`Catalyst_Type × idade`) | só se H3 der significativo |

**Modelo de Yield (P2)**, em duas etapas:

- **Etapa A (estado → eficiência):** `Health ~ f(Vibração)`, com a feature `Vib_Degradado = vib > 6,5`. É o que indica a necessidade de manutenção e o que os cenários usam para projetar o Health.
- **Etapa B:** `log(Yield) ~ log_Flow + log_Health`.
- **Visão alternativa:** "Yield sem Health" (só vazão + vibração), para mostrar quanto do estado mecânico, sozinho, prevê perda.

**Removidas, com evidência de ablação na CV:**

- `Delta_Temp`, `Severidade`, `Carga_por_Abertura`: combinam variáveis sem efeito. `Carga_por_Abertura` ainda cria uma **alavanca espúria** na válvula dentro do otimizador.
- `Idade_Norm`: redundante (corr = 1 com a idade).
- `Vibration` no modelo de EI: efeito totalmente mediado pelo Health.
- `Ambient`, one-hot de `Unit`/`Catalyst`, features temporais: sem efeito (H4/H6).

**Leakage:** `Electricity`, `Gas`, `Steam` e o outro target ficam fora dos dois modelos.

## Etapa 3 — Machine Learning (seções 4.2 e 5)

- **4.2 — demonstração de leakage:** Ridge **com** consumo + Yield (R²≈1) × modelo honesto (0,71) × teto oráculo (0,713). Gráfico de barras lado a lado.
- **Dois modelos por target:** Ridge log-log (interpretável, fornece a elasticidade e permite LP) × `HistGradientBoostingRegressor`.
- **Linhas extras na tabela:** "baseline atual (12 features)" e "com leakage".
- **Métricas:** R², RMSE, MAE, MAPE, com CV KFold(5) e TimeSeriesSplit(5) em média ± desvio, mais o holdout temporal de 20%.
- **5.2 — efeitos:**
  - coeficientes do OLS (statsmodels) com IC 95% para **todos** os controláveis, provando que T/P/V ≈ 0;
  - importância por permutação no HGB como confirmação;
  - tabela "efeito de +10% em cada controlável sobre EI".
- **5.3 — escolha:** Ridge log-log, porque atinge o teto teórico, é suave para o otimizador e dá elasticidades legíveis. O HGB fica como verificação cruzada na solução ótima.

## Etapa 4 — Otimização (seção 6)

- **Novas premissas em `config.py`:**
  - `PRECO_FEEDSTOCK_M3`, calibrado para o feedstock representar ~70% da receita média (típico do setor); ≈ 71 R$/m³;
  - `CUSTO_TROCA_CATALISADOR`, `DOWNTIME_TROCA_CATALISADOR_H`, `TAXA_DEGRADACAO_VIBRACAO` (mm/s por dia).
- **Rota A — LP com `linprog`:** com o modelo log-log, Y é linear em F para um Health fixo e a energia não depende de F, então a margem é linear em F. Formulação:
  - `max (p·k·H − c_f)·F`
  - sujeito a `k·H·F ≥ Y_min` e `F ∈ [lb, ub]`.
  - Reportar o preço-sombra da capacidade.
- **Rota B — SLSQP com o HGB:** comparar com a Rota A.
- **T/P/V:** efeito nulo, então ficam fixos na operação atual (critério de menor perturbação). Isso é dito explicitamente, sem fingir que são uma recomendação.
- **Viabilidade por regime:** em regime degradado, Y_min fica inviável → sinaliza manutenção obrigatória (nova regra de negócio).
- **Sensibilidade:** Y_min em 75/80/85, preço do feedstock, Health.
- **6.6 — checklist de sanidade:** solução no limite (esperado pela estrutura, explicado), distância até o ponto histórico mais próximo, concordância Ridge × HGB.

## Etapa 5 — Cenários de manutenção (seção 7), em 2 tipos

- **Mecânica:** reduz a vibração abaixo de 6,5, o que faz E[Health] ir de 0,68 a 0,85 e a produção subir ~25%.
  - Custo: `CUSTO_MANUTENCAO_PROGRAMADA` + receita perdida no downtime.
  - `P(falha)`: logística na vibração, ancorada no limiar de 6,5 e nas zonas da ISO 10816. É **premissa** e fica marcada como tal.
- **Catalisador:** custo de degradação `k = 80,74 · 0,035 · 0,5` R$ por leitura de 4 h, por dia de idade.
  - Idade ótima de troca: `T* = √(2·C_troca / k_dia)`, pelo modelo clássico de substituição com degradação linear.
  - Isso substitui o limite atual de 400 dias, que nunca dispara (a idade máxima nos dados é 364).
- **S1/S2/S3 por tipo e combinado**, sempre re-otimizando a operação com o estado de cada cenário.
- **Projeção do estado no tempo:** é premissa, porque os dados são cross-sectional (autocorrelação ≈ 0). Isso fica dito.
- **7.3 — varredura de postergação de 0 a 60 dias:** curva de custo esperado com o ponto de indiferença (slide principal).

## Etapa 6 — Automação (seção 8)

- **Limiar ótimo:** `p* = C_FP / C_FN ≈ 120k / 800k = 0,15`, comparado ao ingênuo 0,5 (gráfico de custo × limiar).
- **Gates completos em `gate_automacao()`:**
  - regime degradado;
  - fora da faixa de treino (OOD);
  - **discordância Ridge × HGB acima de X%**;
  - **drift**: teste KS das entradas contra a distribuição de treino.
- **Recomendação:** L2 para a vazão (reversível) e L1 para as ordens de manutenção.

## Etapa 7 — Regras de negócio e restrições revisadas (após os modelos)

Atualizar a seção 1.3 do notebook, `config.py` e `docs/01_entendimento_negocio.md`:

- **R3 — vibração crítica:** 6,0 → **6,5** (limiar dos dados). O alerta de 4,5 é mantido (fronteira B/C da ISO 10816).
- **R4 — Health < 0,70:** reinterpretar. Nos dados, isso **só acontece** em regime degradado, então é degradação certa. A faixa 0,70–0,80 é ambígua e deve ser resolvida pela vibração. Documentar que o "Sensor_Health" se comporta como fator de eficiência do processo, não como confiabilidade do sensor.
- **Nova R5 — manutenção obrigatória:** se Y_min for inviável dentro dos limites seguros.
- **Nova R6 — troca de catalisador:** quando a idade ≥ T*.
- **Limites operacionais:** P5–P95 mantidos. Para T/P/V, os limites existem só por segurança.
- **Documentação:** corrigir a doc 1.2 (sazonalidade refutada) e a tabela de classificação.

## Etapa 8 — Notebook, seções 9–12

- Tabela final de decisão preenchida, comparada com a linha de base (operação média histórica), com o ganho em R$.
- Conclusão com as 6 respostas.
- **Export:** `metadata.json` enriquecido com features por modelo (com justificativa), excluídas (com motivo), achados H1–H8 com os números, métricas completas, coeficientes/elasticidades, T* e premissas.
- Figuras-chave salvas em `app/static/figures/`.

## Etapa 9 — Produto web

- **Nova página `/pipeline`:** uma seção por etapa (Dados → EDA → FE → Leakage → Modelos → Otimização → Cenários → Decisão → Automação), cada uma com *o que foi feito, a evidência numérica e a decisão tomada*, lido de `/api/metadata`.
- **Nova página `/modelos`:**
  - tabela comparativa (com as linhas de leakage e baseline);
  - elasticidades e IC;
  - figuras (degrau do Health, gás × idade, EI × vazão);
  - explicação dos achados;
  - *playground* do endpoint `/api/predict` com exemplo de `curl`.
- **Atualizar `/operacao`:** só a vazão é alavanca; T/P/V aparecem como "sem efeito — mantidos".
- **Atualizar `/manutencao`:** os dois tipos de manutenção, T* do catalisador e a curva de postergação.
- **Novos endpoints em `app/main.py`:** `GET /api/findings`, `POST /api/maintenance/sweep`, `GET /api/catalyst/optimal-age`. `maintenance()` passa a usar `src/decisao.py`, e o reset arbitrário `Health=0,98` sai (o Health é projetado a partir da vibração).
- **Navegação:** links para as páginas novas em todas as páginas; estilo de `app/static/app.css` reaproveitado.

## Arquivos críticos

- **Criar:** `src/features.py`, `src/decisao.py`, `app/static/pipeline.html`, `app/static/modelos.html`.
- **Editar:** `notebooks/01_pipeline_completo.ipynb` (seções 1.3, 2–12), `src/config.py`, `app/main.py`, `app/static/{index,operacao,manutencao}.html`, `app/static/app.js`, `scripts/bootstrap_models.py`, `docs/01_entendimento_negocio.md`, `docs/02_dicionario_dados.md`, `README.md`.

## Ordem de execução (prazo é amanhã)

1. Etapas 0–3: código compartilhado, EDA, features, modelos.
2. Etapa 8: export dos modelos. A partir daqui o app já reflete os modelos novos.
3. Etapas 4–6: otimização, cenários, automação.
4. Etapa 7: revisão das regras e da documentação.
5. Etapa 9: páginas web.

## Verificação

1. `jupyter nbconvert --to notebook --execute notebooks/01_pipeline_completo.ipynb`: roda sem erro e sem TODO restante (`grep TODO`).
2. Conferir na saída:
   - honesto ≈ 0,71 contra o teto de 0,713;
   - leakage ≈ 1,0;
   - elasticidade da vazão ≈ −1;
   - IC de T/P/V contém 0;
   - `linprog` e SLSQP concordam;
   - a curva de postergação tem mínimo;
   - T* está dentro de [1, 364].
3. `uvicorn app.main:app` e depois chamar `/api/health`, `/api/metadata`, `/api/predict`, `/api/optimize`, `/api/maintenance`, `/api/maintenance/sweep`, `/api/catalyst/optimal-age`, `/api/findings`, `/api/decision`.
   - Casos obrigatórios: vibração 7,5 (deve recomendar manutenção mecânica e L0/L1) e vibração 3 com catalisador de 300 dias (deve recomendar troca de catalisador).
4. Abrir `/`, `/pipeline`, `/modelos`, `/operacao` e `/manutencao` no navegador, com o console sem erros.
5. `python scripts/bootstrap_models.py` continua funcionando com os módulos compartilhados.
