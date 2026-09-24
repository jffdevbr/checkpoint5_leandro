# Documentação — Pipeline e API

Resumo de **como cada entrega do enunciado ([instrucoes.md](instrucoes.md)) foi feita**, com a tecnologia usada, o método e o que ele significa para o negócio.
O detalhe completo está no notebook [01_pipeline_completo.ipynb](../notebooks/01_pipeline_completo.ipynb). Os números abaixo vêm de [models/metadata.json](../models/metadata.json).

> **Premissas econômicas.** Preços e custos (produto R$ 2.800/t, feedstock R$ 71,4/m³, manutenção R$ 120 mil, falha R$ 800 mil etc.) **não vêm do dataset**. O grupo assumiu esses valores e eles estão centralizados em [src/config.py](../src/config.py). Todo valor em R$ depende deles.

---

## Visão geral

| Camada | Tecnologia | Papel |
|---|---|---|
| Dados e EDA | pandas, numpy, matplotlib/seaborn | leitura, testes de hipótese, gráficos |
| Machine Learning | scikit-learn (Ridge, HistGradientBoosting), statsmodels (OLS) | prever energia e produção |
| Otimização | `scipy.optimize.linprog` (HiGHS) e SLSQP | escolher a vazão ótima |
| Decisão | [src/decisao.py](../src/decisao.py) | cenários, risco e gate de automação |
| Entrega | FastAPI + Pydantic + uvicorn, joblib, Docker | API e páginas web para o operador |

O código de negócio fica em `src/` e é **o mesmo** no notebook e na API. O notebook treina os modelos e salva os arquivos em `models/`, e a API carrega esses mesmos arquivos. Assim o que foi validado é exatamente o que vai para produção.

---

## 1. Entendimento do negócio

**Objetivo da operação:** produzir o máximo com a menor energia por tonelada, sem deixar o equipamento degradar até quebrar.

| Classe | Variáveis | Quem decide |
|---|---|---|
| Controláveis | `Feedstock_Flow_m3h`, `Reactor_Temp_C`, `Reactor_Pressure_Bar`, `Valve_Opening_Percent` | operador, em tempo real |
| Estado | `Catalyst_Age_Days`, `Sensor_Health_Index`, `Vibration_Level_mm_s` | mudam só com manutenção |
| Externas | `Ambient_Temp_C`, `Unit_Name`, `Catalyst_Type` | fora do controle |
| Consumo (leakage) | `Electricity_MWh`, `Natural_Gas_m3h`, `Steam_Tons_h` | consequência da operação |
| Targets | `Energy_Intensity`, `Product_Yield_Tons` | resultado |

**Restrições e regras de negócio:**

- **R1:** produção ≥ 80 t por janela de 4 h (a média histórica, ou seja, "manter a produção").
- **R2:** setpoints dentro dos percentis 5–95 do histórico, que é a região onde o modelo foi treinado.
- **R3:** vibração acima do limite crítico torna a manutenção obrigatória.
- **R4:** Health do sensor < 0,70 impede a automação, porque o dado de entrada não é confiável.
- **R5:** se a produção mínima fica inviável, a manutenção passa a ser obrigatória.

Detalhes em [01_entendimento_negocio.md](01_entendimento_negocio.md) e [02_dicionario_dados.md](02_dicionario_dados.md).

**Achado central da EDA:** as linhas são independentes (autocorrelação ≈ 0), então o problema é de **regressão**, não de série temporal. Os testes de hipótese mostraram que:

- das variáveis controláveis, **só a vazão de matéria-prima** (`Feedstock_Flow_m3h`, quantos m³/h de insumo entram no reator) altera produção e energia. Temperatura, pressão e válvula não mostraram efeito nos dados (H5);
- a vibração acima de **6,5 mm/s** derruba o Health de ~0,85 para ~0,68 (H1), o que corta ~20% da produção;
- cada dia de idade do catalisador soma 0,5 m³/h de gás (H2).

---

## 2. Machine Learning

**Target leakage.** Descobrimos que `Energy_Intensity = (3,6·Electricity + 0,035·Gas) / Yield` exatamente (R² = 1,0). Usar o consumo como feature dá R² = **0,999**, mas o modelo não serve para otimizar: para prever a energia, ele já precisaria conhecer a energia. Sem leakage, o R² honesto é **0,711**, praticamente igual ao teto teórico (0,713).

**Feature engineering.** Usamos poucas features, e cada uma tem motivo:

- `log_Flow` e `log_Health`: no modelo log-log, o coeficiente é a elasticidade;
- `Catalyst_Age_Days`: afeta o consumo de gás;
- `Vib_Degradado`: indica vibração acima de 6,5 mm/s.

**Comparação (validação cruzada KFold(5); também validado com TimeSeriesSplit e holdout):**

| Target | Modelo | R² | RMSE | MAPE |
|---|---|---|---|---|
| Energy | **Ridge log-log** | **0,711** | 0,336 | 9,5% |
| Energy | HistGradientBoosting | 0,695 | 0,345 | 9,8% |
| Yield | **Ridge log-log** | **≈ 1,000** | 0,002 | 0,002% |
| Yield | HistGradientBoosting | 0,999 | 0,443 | 0,39% |
| Yield | Ridge log-log **sem Health** (cenários futuros) | 0,657 | 8,25 | 8,9% |

> **Por que o yield tem R² ≈ 1?** Não é leakage. O modelo de yield também não usa os consumos nem `Energy_Intensity`. Uma versão treinada com essas colunas teve R² = 0,998, abaixo do modelo honesto, então o vazamento nem ajudaria. O R² perfeito vem de uma fórmula exata do dataset, `Yield = 0,18 × Vazão × Health`: com logaritmo, ela fica igual à forma do Ridge log-log, e o modelo reencontra a fórmula (intercepto = ln 0,18; elasticidades = 1,0). É um sinal de que os **dados são sintéticos**, não um mérito do modelo.
>
> **Health é leakage?** Aqui não. Ele é uma variável de estado do equipamento, medida junto com a leitura, e não é calculado a partir da produção. Mas o nome engana: ele não mede a saúde do sensor, funciona como um **fator de eficiência** que multiplica a produção. Numa planta real, se esse índice fosse calculado como produção real ÷ teórica, seria leakage.
>
> **Desempenho realista:** o R² ≈ 1 só vale para a operação do momento, com o Health já medido. Para o futuro (cenários de manutenção), o Health é desconhecido. Por isso o pipeline primeiro estima o Health pela vibração e depois prevê a produção. Nessa situação, o yield tem **R² = 0,657** e **MAPE = 8,9%**.

**Escolhido: Ridge log-log** nos dois targets, por quatro motivos:

1. **Erro:** é o melhor modelo nas três validações.
2. **Suavidade:** não tem os degraus da árvore, que atrapalham o otimizador.
3. **Formulação:** vira um problema linear, que cabe no `linprog`.
4. **Interpretação:** o coeficiente é lido direto pelo operador. Por exemplo, +10% de vazão dá ≈ −9% de intensidade energética e +10% de produção.

O HGB continua no pipeline como **verificação cruzada**. Se os dois modelos divergirem, a solução fica suspeita.

---

## 3. Otimização

- **Variável de decisão:** a vazão de matéria-prima `F` (`Feedstock_Flow_m3h`). T, P e válvula ficam fixas no valor atual: como não mostraram efeito nos dados (H5), qualquer valor dentro da faixa dá o mesmo resultado. Numa planta real, temperatura e pressão afetam a reação, então essa é uma característica deste dataset e não uma regra geral.
- **Função objetivo** (margem por janela de 4 h):

```
max  m(F) = p·Ŷ(F) − c_E·ÊI(F)·Ŷ(F) − 4·c_f·F
s.a. Ŷ(F) ≥ 80 t                        (produção mínima)
     451,9 ≤ F ≤ 648,8 m³/h             (P5–P95 do histórico)
```

Com o modelo log-log, `Ŷ = k·H·F` é linear em F e a energia total quase não depende de F. Então a margem é **linear**, e usamos `linprog` com o método HiGHS (Rota A, erro de linearização de 0,007%). Para conferir, rodamos SLSQP no problema não linear (Rota B) e ele chegou na mesma resposta.

**Solução:** a vazão sobe de 554,7 para **648,8 m³/h**, e o resto não muda.

| | Atual | Ótimo |
|---|---|---|
| Produção (t/janela) | 84,6 | **99,0** |
| Intensidade energética | 2,62 | **2,25** (−14%) |
| Margem (R$/janela) | 60,6 mil | **73,9 mil** (+22%) |

**Interpretação:**

- **Por que aumentar a vazão reduz a intensidade:** a energia gasta na janela é praticamente fixa. Com mais vazão, ela é **dividida por mais toneladas**.
- **Por que o ótimo fica no limite:** cada m³/h a mais rende R$ 141 por janela (preço-sombra da capacidade).
- **Quando a operação sozinha não resolve:** se o Health cai abaixo de 0,685, nenhuma vazão segura entrega 80 t. A partir daí a solução passa a ser manutenção.

---

## 4. Decisão de manutenção

Existem dois tipos de manutenção, com lógicas diferentes:

- **Mecânica:** reduz a vibração, recupera o Health e evita falha;
- **Troca de catalisador:** reduz o consumo de gás.

Em cada cenário a operação é **re-otimizada dia a dia** com o estado projetado. Os cenários são comparados pelo **resultado esperado por dia**, para que janelas de tamanhos diferentes fiquem comparáveis.

Estado atual: vibração 5,1 mm/s, Health 0,85, catalisador com 128 dias.

| Cenário | Produção | Energia | Condição | P(falha) | Resultado/dia |
|---|---|---|---|---|---|
| S1 — Sem manutenção | cai para 79,6 t (52 dias abaixo do mínimo) | EI ≈ +25% | cruza 6,5 mm/s em ~28 d | **81%** | R$ 226 mil |
| S2 — Mecânica imediata | 99,4 t | normal | vibração volta a 2,5 mm/s | 7% | R$ 442 mil |
| **S3 — Mecânica em 28 dias** | 99,4 t | normal | faz a manutenção logo antes do limiar | 14% | **R$ 443 mil** |

**Leitura:** postergar ~28 dias aproveita a vida útil que ainda resta, sem entrar no regime degradado. **Adiar mais de 30 dias fica pior do que fazer hoje.** Trocar o catalisador agora não compensa: a idade ótima de troca é ~284 dias (ou ~168 dias se a troca for feita junto de uma parada mecânica).

**Premissas assumidas:** a vibração cresce 0,05 mm/s por dia, e o risco de falha segue as zonas da ISO 10816. O dataset não permite estimar essas duas coisas, e a sensibilidade a elas está no notebook.

---

## 5. Automação da decisão

| Benefícios | Riscos |
|---|---|
| Reação em minutos, 24/7, em várias plantas ao mesmo tempo | **Falso positivo:** parada desnecessária (R$ 120 mil + margem perdida) |
| Menos intervenção manual e menos downtime não programado | **Falso negativo:** quebra (R$ 800 mil + risco de segurança), o erro mais caro |
| Ajuste contínuo da vazão (+R$ 13 mil a cada 4 h) | Erro do modelo, sensor ruim, mudança de regime da planta |

**Recomendação: automação assimétrica**, graduada pela reversibilidade de cada decisão:

- **L2 para setpoints:** o sistema ajusta a vazão sozinho, porque a ação é reversível e barata.
- **L1 para manutenção:** o sistema recomenda e um humano aprova, porque a ação é irreversível, cara e envolve segurança.
- **Gate automático para L0** (humano decide tudo) quando:
  - o Health está abaixo de 0,70;
  - a vibração está acima do limite crítico;
  - o ponto está fora da faixa de treino.

O nível de automação não é fixo: ele depende da **confiança do modelo naquele ponto**. Essa regra está implementada em `gate_automacao`.

---

## 6. Pipeline final

```
Dados → EDA → Feature Eng. → checagem de leakage → Ridge (Energia, Yield) + HGB (verificação)
  → Previsões → Cenários S1/S2/S3 → Otimização LP em cada estado → Custo + risco
  → Decisão de manutenção → Recomendação → Gate de confiança → L2 automático | L1/L0 humano
                  ↑__________ manutenção muda o estado → muda a operação ótima ________|
```

A seta de volta é o ponto principal: otimização e manutenção estão **acopladas**.

---

## 7. Resultado final

| Item | Valor |
|---|---|
| Feedstock Flow | **648,8 m³/h** (antes 554,7) |
| Reactor Temperature | 826,8 °C (mantida) |
| Reactor Pressure | 31,9 bar (mantida) |
| Valve Opening | 48,7 % (mantida) |
| Expected Yield | 99,0 t por janela de 4 h |
| Energy Intensity | 2,25 |
| Energy Cost | R$ 17,96 mil por janela |
| Maintenance | **Sim**: mecânica programada em 28 dias; catalisador não |
| Maintenance Cost | R$ 120 mil de serviço + ~R$ 450 mil de margem perdida (24 h de parada) |
| Total Cost | R$ 203,3 mil por janela (feedstock + energia) + R$ 570 mil por ciclo de manutenção |
| Nível de automação | L2 para setpoints e L1 para manutenção (hoje L1, porque a vibração está acima de 4,5 mm/s) |

---

## 8. Conclusão

- **Configuração recomendada:** vazão no limite operacional (648,8 m³/h), mantendo T, P e válvula.
- **Manutenção:** sim. A mecânica deve ser programada em ~28 dias, ou imediatamente se a vibração passar de 6,5 mm/s ou a produção mínima ficar inviável. Trocar o catalisador só a partir de ~284 dias de idade.
- **Impacto econômico:**
  - otimização: +R$ 13,3 mil a cada 4 h, ≈ **R$ 2,4 milhões por mês**;
  - manutenção: ≈ **+R$ 217 mil por dia** em relação a não fazer, além de evitar 81% de chance de uma falha de R$ 800 mil.
- **Riscos:**
  - premissas de preço e custo assumidas pelo grupo;
  - curva de falha não calibrada com falhas reais;
  - vazão ótima no limite do histórico;
  - dados observacionais (correlação não garante causa).
- **Automatizar?** Parcialmente: setpoints de forma automática, manutenção com aprovação humana.
- **Dados que faltam para produção:**
  - histórico real de falhas e ordens de manutenção;
  - preços reais de energia, feedstock e produto;
  - limites oficiais de segurança da planta (HAZOP);
  - série temporal real da vibração;
  - testes controlados de setpoints.

---

## API

A API é construída com **FastAPI**. Ela carrega os modelos `.pkl` e o `metadata.json` exportados pela seção 12 do notebook. Se os modelos não existem, ela responde 503 com instruções em vez de quebrar. A documentação interativa fica em `/docs` (Swagger).

| Rota | Entrada | Saída |
|---|---|---|
| `POST /api/predict` | setpoints + contexto | intensidade energética, produção, custos e margem |
| `POST /api/optimize` | contexto + produção mínima | vazão ótima (LP) + nível de automação |
| `POST /api/maintenance` | contexto + dias de postergação | S1/S2/S3 comparados + recomendação |
| `GET /api/decision` | estado do equipamento | tabela final de decisão (entrega 7) |
| `GET /api/metadata` | — | variáveis, limites, premissas e métricas |
| `GET /api/health` | — | status da API e dos modelos |

**Páginas para o operador:**

- `/`: visão geral e achados;
- `/operacao`: simulador de setpoints com botão de otimizar;
- `/manutencao`: cenários, risco e nível de automação.

**Como rodar:**

```powershell
docker compose up --build        # API em :8000 e JupyterLab em :8888
# ou, sem Docker:
uvicorn app.main:app --reload
```

> **Limitação conhecida.** O `/api/maintenance` usa uma versão simplificada dos cenários: horizonte fixo de 30 dias, e o S2 reinicia o Health em 0,98. Por isso ele pode recomendar manutenção **imediata**, enquanto a análise do notebook (seção 7) recomenda **programar em 28 dias**. A recomendação oficial é a do notebook.
