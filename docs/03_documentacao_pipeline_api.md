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

**Os dois modelos:**

- **Ridge log-log:** uma regressão linear com logaritmo na entrada e na saída. Cada coeficiente é uma **elasticidade**: "+1% nesta variável → +b% no target". A parte **Ridge** acrescenta uma pequena penalidade (`alpha = 1`) que impede coeficientes exagerados. Ele é simples, suave e fácil de explicar. Em compensação, só capta as relações que colocamos nele.
- **HistGradientBoosting (HGB):** um conjunto de centenas de **árvores de decisão** pequenas, treinadas em sequência. Cada árvore corrige o erro das anteriores. "Hist" quer dizer que os valores são agrupados em faixas, o que deixa o treino rápido. Ele encontra sozinho relações não lineares e interações entre variáveis. Em compensação, a previsão muda em degraus e é difícil de explicar.

Os dois são do scikit-learn. O Ridge é o modelo usado no pipeline. O HGB serve de **contraprova**: se nem um modelo flexível acha sinal além do Ridge, é porque não há sinal a mais nos dados.

**Features de cada modelo** (nenhum usa os consumos nem o target do outro):

| Modelo | Target | Features |
|---|---|---|
| Ridge log-log — energia | `log(Energy_Intensity)` | `log_Flow`, `log_Health`, `Catalyst_Age_Days` |
| Ridge log-log — yield | `log(Product_Yield_Tons)` | `log_Flow`, `log_Health` |
| Ridge — Health (etapa A dos cenários) | `Sensor_Health_Index` | `Vib_Degradado` |
| Ridge log-log — yield sem Health | `log(Product_Yield_Tons)` | `log_Flow`, `Vib_Degradado` |
| HGB — energia e yield | o target original | as 8 variáveis brutas honestas: vazão, temperatura, pressão, válvula, idade do catalisador, Health, vibração, temperatura ambiente |

- `log_Flow` e `log_Health`: logaritmo da vazão e do Health; no log-log, o coeficiente vira elasticidade.
- `Catalyst_Age_Days`: catalisador mais velho consome mais gás (H2), por isso só entra no modelo de energia.
- `Vib_Degradado`: 1 se a vibração passa de 6,5 mm/s, 0 se não. Marca o regime em que o Health cai (H1).
- Ficaram de fora do Ridge: temperatura, pressão, válvula e ambiente, que não têm efeito (H4/H5), e planta e tipo de catalisador, que não fazem diferença (H3/H6). O HGB recebe temperatura, pressão, válvula e ambiente de propósito, para confirmar que elas não têm efeito.

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

O HGB (HistGradientBoosting) continua no pipeline como **verificação cruzada**. Se os dois modelos divergirem, a solução fica suspeita.

---

## 3. Otimização

**O que é, em palavras simples.** Otimizar é deixar o computador testar, de forma inteligente, todas as combinações possíveis de ajustes da planta e escolher **a que dá o melhor resultado sem quebrar nenhuma regra**. Parece com um GPS: você diz o destino (o objetivo) e as regras (evitar pedágio, não passar de 80 km/h), e ele acha a melhor rota. Aqui:

- **O que queremos:** a maior **margem**, ou seja, o que sobra da receita depois de pagar matéria-prima e energia.
- **Regras que precisam ser respeitadas:** produzir pelo menos 80 t por janela de 4 h, sem tirar a planta da faixa em que ela já operou.
- **O que o computador pode mexer:** os ajustes do operador.

**Para que serve.** Os modelos de ML só respondem "se eu ajustar assim, o que acontece?". A otimização inverte a pergunta: **"qual ajuste devo usar?"**. É ela que transforma uma previsão numa recomendação para o operador.

**O que pode ser ajustado.** Na prática, só a **vazão de matéria-prima** (`Feedstock_Flow_m3h`). Temperatura do reator, pressão do reator e abertura da válvula não mostraram efeito nos dados (H5): qualquer valor dentro da faixa dá o mesmo resultado. Por isso elas **ficam no valor atual**, sem mexer no que não traz ganho. Numa planta real, temperatura e pressão afetam a reação; essa é uma característica deste dataset, não uma regra geral.

**A conta que está sendo maximizada** (margem de uma janela de 4 h):

```
margem = preço do produto × produção prevista
       − custo da energia
       − custo da matéria-prima

produção prevista      = modelo Ridge de yield           (depende da vazão e do Health)
custo da energia       = R$ 80,74 × intensidade energética prevista × produção prevista
custo da matéria-prima = 4 h × R$ 71,4/m³ × vazão

regras:  produção prevista ≥ 80 t
         451,9 m³/h ≤ vazão ≤ 648,8 m³/h    (faixa de 90% do histórico: percentis 5 a 95)
```

Os limites da vazão vêm do próprio histórico. Fora dessa faixa o modelo estaria "chutando", porque nunca viu a planta operar ali.

**Como o computador resolve (duas rotas, para conferir uma com a outra):**

- **Rota A: programação linear, com `linprog` (biblioteca SciPy, método HiGHS).** Programação linear é o caso mais simples de otimização: todos os ganhos e custos são proporcionais à variável, como "cada m³/h a mais rende R$ X". Nesse caso existe um método exato e rapidíssimo para achar a melhor resposta. O nosso problema cabe aqui porque:
  - a produção é proporcional à vazão (`produção = 0,18 × vazão × Health`);
  - a energia total gasta na janela quase não muda com a vazão.
  
  Então a margem vira "ganho por m³/h × vazão". O erro dessa simplificação é de só 0,007%. HiGHS é o motor de cálculo, de código aberto, que o `linprog` usa por dentro.
- **Rota B: SLSQP, no problema completo, sem simplificar.** SLSQP (*Sequential Least Squares Programming*, ou "programação sequencial por mínimos quadrados") é um método para problemas **não lineares** com regras. Ele parte de um ponto, olha para que lado a margem cresce, dá um passo nessa direção e repete até não conseguir melhorar mais. A cada passo, ele troca o problema difícil por uma aproximação mais simples, que dá para resolver. Serviu para confirmar que a simplificação da Rota A não mudou a resposta, e as duas chegaram ao **mesmo resultado**.
  - Com o HGB no lugar do Ridge, o SLSQP parou antes do ótimo, em 545 m³/h. Os "degraus" das árvores enganam o método sobre a direção de melhora, e é por isso que o Ridge é o modelo usado na otimização.

**Solução:** a vazão sobe de 554,7 para **648,8 m³/h**. Temperatura do reator, pressão do reator e abertura da válvula continuam como estão.

| Por janela de 4 h | Atual | Ótimo |
|---|---|---|
| Produção | 84,6 t | **99,0 t** |
| Intensidade energética | 2,62 | **2,25** (−14%) |
| Margem | R$ 60,6 mil | **R$ 73,9 mil** (+22%) |

**Interpretação:**

- **Por que produzir mais gasta menos energia por tonelada:** a energia gasta na janela é praticamente fixa. Com mais vazão, ela é **dividida por mais toneladas**. É como um ônibus: o combustível da viagem é quase o mesmo, então quanto mais passageiros, menor o custo por passageiro.
- **Por que a resposta é "o máximo permitido":** cada m³/h a mais rende R$ 141 de margem por janela. Esse valor se chama **preço-sombra**: quanto a margem aumentaria se o limite de vazão fosse 1 m³/h maior. Enquanto ele for positivo, vale ir até o limite. Ele também diz à engenharia quanto valeria ampliar a capacidade, se isso for seguro.
- **Quando ajustar a operação não basta:** se o Health cai abaixo de 0,685, nenhuma vazão permitida chega a 80 t. A partir daí o problema deixa de ser de ajuste e passa a ser de **manutenção** (seção 4).

---

## 4. Decisão de manutenção

Existem dois tipos de manutenção, com lógicas diferentes:

- **Mecânica:** reduz a vibração, recupera o Health e evita falha;
- **Troca de catalisador:** reduz o consumo de gás.

Em cada cenário a operação é **re-otimizada dia a dia** com o estado projetado. Os cenários são comparados pelo **resultado esperado por dia**, para que janelas de tamanhos diferentes fiquem comparáveis.

**Estado atual:** vibração 5,1 mm/s, Health 0,85, catalisador com 128 dias. Esses valores não são de uma planta específica: são a **mediana das últimas 24 leituras do dataset** (20/07 a 24/07/2024, cerca de 4 dias), que misturam as três unidades (12 de metanol, 8 de amônia e 4 de eteno). Juntar as plantas é aceitável porque a EDA mostrou que elas não se comportam como populações diferentes (H6). Usamos a mediana, e não a última leitura, porque as leituras variam muito de uma para outra: a vibração foi de 0,7 a 8,2 mm/s nesses 4 dias. Em produção, o estado viria dos sensores **da unidade** que está sendo avaliada, e a API já recebe esses valores como entrada.

| Cenário | Produção | Energia | Condição | P(falha) | Resultado/dia |
|---|---|---|---|---|---|
| S1 — Sem manutenção | cai para 79,6 t (52 dias abaixo do mínimo) | EI ≈ +25% | cruza 6,5 mm/s em ~28 d | **81%** | R$ 226 mil |
| S2 — Mecânica imediata | 99,4 t | normal | vibração volta a 2,5 mm/s | 7% | R$ 442 mil |
| **S3 — Mecânica em 28 dias** | 99,4 t | normal | faz a manutenção logo antes do limiar | 14% | **R$ 443 mil** |

**Leitura:** postergar ~28 dias aproveita a vida útil que ainda resta, sem entrar no regime degradado. **Adiar mais de 30 dias fica pior do que fazer hoje.**

**De quanto em quanto tempo fazer a manutenção mecânica (vibração):**

- **A primeira:** em ~28 dias. A vibração está em 5,1 mm/s e sobe 0,05 mm/s por dia, então passa do limite de 6,5 mm/s em (6,5 − 5,1) ÷ 0,05 ≈ 28 dias.
- **As seguintes:** a cada **~80 dias**. Depois da manutenção, a vibração volta a 2,5 mm/s e leva (6,5 − 2,5) ÷ 0,05 = 80 dias para chegar de novo ao limite.
- **A regra geral** é fazer a manutenção logo antes de a vibração cruzar 6,5 mm/s. O calendário fixo é só uma consequência da taxa assumida, e na prática o gatilho deve ser a **vibração medida**, não a data.
- **Sensibilidade a essa taxa:** se a vibração subir na metade da velocidade (0,025 mm/s/dia), o ciclo passa a 160 dias; se subir no dobro (0,1), cai para 40 dias. A taxa real precisa ser medida na planta.

**Catalisador:** trocar agora não compensa: a idade ótima de troca é ~284 dias (ou ~168 dias se a troca for feita junto de uma parada mecânica).

**Premissas assumidas:** a vibração cresce 0,05 mm/s por dia, e o risco de falha segue as zonas da ISO 10816. O dataset não permite estimar essas duas coisas, e a sensibilidade a elas está no notebook.

**O que é a ISO 10816.** É uma norma internacional usada na indústria para avaliar se o nível de vibração de uma máquina (bombas, compressores, motores, turbinas) é aceitável. A vibração é medida na carcaça da máquina, em mm/s, e classificada em quatro zonas. Os limites dependem do porte da máquina e da base onde ela está montada. Usamos os de máquinas grandes (acima de 300 kW) em base rígida:

| Zona | Vibração | Significado na norma | Uso no projeto |
|---|---|---|---|
| A | até ~2,3 mm/s | máquina nova ou recém-reparada | vibração após a manutenção: 2,5 mm/s (percentil 25 dos dados) |
| B | 2,3 a 4,5 mm/s | aceitável para operar sem restrição | risco de falha baixo |
| C | 4,5 a 7,1 mm/s | insatisfatória: operar só por tempo limitado, planejar intervenção | 4,5 mm/s = 1% de chance de falha em 30 dias |
| D | acima de 7,1 mm/s | severa: risco de dano à máquina | 7,1 mm/s = 30% de chance de falha em 30 dias |

A norma dá **as fronteiras das zonas, mas não probabilidades de falha**. Os valores de 1% e 30% são premissas do grupo, usadas para desenhar uma curva de risco que cresce com a vibração. Em produção, essa curva deve ser recalibrada com o histórico real de falhas. A ISO 10816 hoje está sendo substituída pela série **ISO 20816**, que mantém a mesma lógica de zonas.

O limite de **6,5 mm/s**, que dispara a manutenção, não vem da norma. Ele vem **dos dados** (H1): é a vibração a partir da qual o Health cai e a produção despenca. Fica dentro da zona C, onde a própria norma já recomenda planejar a intervenção.

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

- **Configuração recomendada:** vazão no limite operacional (648,8 m³/h), mantendo temperatura do reator, pressão do reator e abertura da válvula.
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
