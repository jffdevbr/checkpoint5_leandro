# 1. Entendimento do Negócio — petrochemical_advanced_data.csv

## 1.0 Contexto: o que essa indústria produz e o que está sendo medido

> **Nota de vocabulário:** "planta", aqui, não é o ser vivo — é o termo da indústria (do inglês *plant*) para uma **instalação/fábrica**: o conjunto de reatores, tubulações e equipamentos que formam uma unidade de produção. "Planta petroquímica" = fábrica petroquímica.

As 3 unidades do dataset são plantas químicas que transformam uma matéria-prima em um produto, através de uma **reação química dentro de um reator**. Cada uma produz uma coisa diferente:

- **`Ethylene_Plant_01`** — produz **eteno (etileno)**, uma matéria-prima básica usada para fabricar plásticos (como o PVC e o polietileno de sacolas/embalagens). O processo típico é o **"craqueamento"**: aquece-se um hidrocarboneto (nafta ou gás) a temperatura muito alta até as moléculas grandes se "quebrarem" em moléculas menores.
- **`Ammonia_Unit_02`** — produz **amônia**, usada principalmente para fabricar fertilizantes. É feita reagindo nitrogênio (do ar) com hidrogênio (do gás natural) sob alta pressão, na presença de um catalisador.
- **`Methanol_Complex_03`** — produz **metanol**, um álcool usado como matéria-prima química e combustível, feito a partir de gás natural (via gás de síntese) reagindo sob pressão com um catalisador.

Apesar de serem processos quimicamente diferentes, todos seguem a mesma lógica operacional, que é o que o dataset está medindo:

```
Matéria-prima (Feedstock)  →  REATOR (temperatura + pressão + catalisador)  →  Produto + calor/energia consumida
```

**O que cada grupo de colunas representa nesse processo:**

- **Matéria-prima entrando:** `Feedstock_Flow_m3h` — quanto de insumo (gás/nafta) está sendo alimentado ao reator por hora. É uma torneira que o operador controla.
- **Condições do reator (as "receitas" da reação):** `Reactor_Temp_C`, `Reactor_Pressure_Bar`, `Valve_Opening_Percent` — o operador ajusta esses parâmetros para a reação acontecer de forma eficiente. Temperatura e pressão erradas geram menos produto, gastam mais energia ou até danificam o equipamento.
- **O catalisador:** `Catalyst_Type`, `Catalyst_Age_Days` — o catalisador é uma substância que fica dentro do reator e "facilita" a reação química sem ser consumida por ela (na teoria), mas na prática **se desgasta com o uso** (perde eficiência com a idade) e precisa ser trocado periodicamente — essa troca é um evento de manutenção.
- **Saúde do equipamento (bombas, compressores, sensores):** `Sensor_Health_Index`, `Vibration_Level_mm_s` — não medem a reação química, medem a condição física do maquinário. Vibração alta e saúde de sensor baixa são sinais de desgaste mecânico, como um carro que começa a fazer barulho antes de quebrar — é o que embasa a decisão de manutenção.
- **Energia consumida para a reação acontecer:** `Electricity_MWh`, `Natural_Gas_m3h`, `Steam_Tons_h` — essas reações em geral precisam de calor/pressão, o que consome eletricidade (motores/compressores), gás natural (queimado para gerar calor) e vapor (usado para aquecer ou como parte do processo).
- **Condição externa:** `Ambient_Temp_C` — a temperatura do ambiente ao redor da planta, que afeta a eficiência de trocadores de calor e resfriamento (não é algo que a planta controla).
- **O resultado da operação (os dois targets do trabalho):**
  - `Product_Yield_Tons` — quantas toneladas de produto (eteno/amônia/metanol) saíram do reator naquela janela de 4h. É literalmente **quanto a planta produziu**.
  - `Energy_Intensity` — quanta energia foi gasta **por tonelada produzida**. É a métrica de eficiência: quanto menor, melhor (mais produto com menos energia). É por isso que ela se relaciona tão fortemente com `Product_Yield_Tons` (ver alerta de leakage na seção 1.5) — ela é, na prática, uma razão entre energia gasta e produção.

**Em resumo, o negócio é:** operar o reator (ajustando vazão, temperatura, pressão, válvula) de forma a produzir o máximo de produto gastando o mínimo de energia por tonelada, sem deixar o equipamento se degradar a ponto de quebrar — e, quando os sinais de desgaste aparecem (vibração alta, sensor de saúde baixo, catalisador velho), decidir se e quando fazer manutenção.

## 1.1 O grão da tabela

Cada **linha** representa **uma leitura operacional pontual (snapshot) de uma unidade industrial, em um determinado instante, a cada 4 horas**.

Evidências:

- 10.000 linhas, 10.000 timestamps **únicos**, cobrindo de `2020-01-01 00:00` a `2024-07-24 12:00` em grade fixa de 4 em 4 horas (10.000 slots de 4h nesse intervalo = 10.000 linhas — bate exatamente).
- Ou seja, **não há duas unidades reportando no mesmo timestamp**: a cada 4h, apenas uma das três plantas (`Ethylene_Plant_01`, `Ammonia_Unit_02`, `Methanol_Complex_03`) aparece no dataset (distribuição quase igual: ~3.300–3.350 linhas cada).
- Não é uma série temporal "paralela" (3 plantas medidas ao mesmo tempo todo o tempo); é mais parecido com um **log de rodízio de medições/relatórios**, onda a cada rodada de coleta apenas uma planta é registrada.

**Chave primária de fato:** `Timestamp` (sozinho já identifica a linha, pois é único). `Unit_Name` é um atributo da linha, não parte de uma chave composta necessária.

## 1.2 Os registros são interdependentes? (série temporal x regressão)

Testei a autocorrelação (lag-1, ordenando por tempo dentro de cada planta) das variáveis mais importantes:

| Variável | Autocorr. lag-1 (Ammonia) | Autocorr. lag-1 (Ethylene) | Autocorr. lag-1 (Methanol) |
|---|---|---|---|
| Energy_Intensity | -0.011 | -0.003 | 0.000 |
| Product_Yield_Tons | -0.029 | -0.029 | 0.001 |
| Sensor_Health_Index | -0.007 | -0.025 | 0.005 |
| Vibration_Level_mm_s | 0.004 | -0.001 | 0.011 |

Todos os valores estão **essencialmente em zero**. Além disso, `Catalyst_Age_Days` "reseta" (valor cai em relação à linha anterior) em ~50% das transições dentro de cada planta — o que indica que o catalisador e sua idade são **sorteados/atribuídos por leitura**, não evoluem de forma contínua e realista ao longo do tempo.

**Conclusão prática:** apesar de existir uma coluna `Timestamp`, **este NÃO é um problema de série temporal clássico** (não há memória/tendência de uma leitura para a próxima que valha a pena modelar com lags, ARIMA, etc.). Trata-se de um **problema de regressão em corte transversal (cross-sectional)**: cada linha é uma observação praticamente independente que relaciona condições operacionais (inputs) a um resultado (Energy_Intensity, Product_Yield_Tons) naquele instante.

Implicações para o pipeline:
- Pode (e deve) fazer split treino/teste aleatório (ou por planta), sem se preocupar com vazamento temporal sequencial linha-a-linha.
- `Timestamp` ainda pode ser útil como **fonte de features derivadas não sequenciais** (mês/estação → explica parte de `Ambient_Temp_C`), mas não como variável de lag.
- Faz sentido tratar cada `Unit_Name` como um contexto/segmento (ou variável categórica) dentro de um único modelo de regressão, e não como 3 séries temporais separadas.

## 1.3 Objetivo da operação

Maximizar a produção (`Product_Yield_Tons`) e minimizar a intensidade energética (`Energy_Intensity`) de cada planta petroquímica, respeitando limites operacionais seguros de reator, vazão e condição do equipamento — e decidir, com base na condição do equipamento (vibração, saúde do sensor, idade do catalisador), se/quando realizar manutenção.

## 1.4 Classificação das variáveis

| Variável | Classificação | Papel no pipeline | Observações |
|---|---|---|---|
| `Timestamp` | Identificador / temporal | Metadado | Único por linha; não é sequencial-dependente (ver 1.2) |
| `Unit_Name` | Identificador / contexto | Segmentação | 3 plantas (Methanol, Ethylene, Ammonia) |
| `Catalyst_Type` | Variável controlável (decisão de longo prazo) | Feature categórica | Escolhida na troca/recarga de catalisador |
| `Catalyst_Age_Days` | Variável de estado | Feature | Indicador de desgaste; 1–364 dias |
| `Sensor_Health_Index` | Variável de estado (condição do equipamento) | Feature / gatilho de manutenção | 0,56–1,00; correlação forte com Yield (+0,78) e Energy_Intensity (-0,66) |
| `Vibration_Level_mm_s` | Variável de estado (condição do equipamento) | Feature / gatilho de manutenção | 0,50–8,50 mm/s; quartil mais alto associado a queda de Sensor_Health_Index |
| `Valve_Opening_Percent` | Variável controlável | Variável de decisão | 20–95% |
| `Feedstock_Flow_m3h` | Variável controlável | Variável de decisão | 341–759 m³/h |
| `Reactor_Temp_C` | Variável controlável (setpoint) | Variável de decisão | 676–966 °C |
| `Reactor_Pressure_Bar` | Variável controlável (setpoint) | Variável de decisão | 20,7–43,6 bar |
| `Ambient_Temp_C` | Variável externa | Feature (não controlável) | -10 a +52 °C |
| `Electricity_MWh` | Consumo de energia (intermediário) | Cuidado com vazamento (ver 1.5) | 15–30 MWh |
| `Natural_Gas_m3h` | Consumo de energia (intermediário) | Cuidado com vazamento (ver 1.5) | 3012–5181 m³/h |
| `Steam_Tons_h` | Consumo de energia (intermediário) | Cuidado com vazamento (ver 1.5) | 40–90 t/h |
| `Product_Yield_Tons` | **Target 1** | Saída a prever | 40,6–134,0 toneladas |
| `Energy_Intensity` | **Target 2** | Saída a prever | 1,33–6,39 (energia por unidade produzida) |

## 1.5 Alerta de possível target leakage (importante para a etapa de ML)

- `Energy_Intensity` tem correlação de **-0,82** com `Product_Yield_Tons` e de **0,41 / 0,33** com `Natural_Gas_m3h` / `Electricity_MWh`.
- Um teste rápido — `Electricity_MWh / Product_Yield_Tons` — já reproduz 82% da variação de `Energy_Intensity` (corr. 0,82). Isso sugere fortemente que `Energy_Intensity` é **calculada a partir do consumo de energia dividido pela produção** (é uma métrica derivada, não uma medição independente).
- **Consequência:** usar `Product_Yield_Tons` como feature para prever `Energy_Intensity` (ou vice-versa), ou usar `Electricity_MWh`/`Natural_Gas_m3h`/`Steam_Tons_h` sem critério em ambos os modelos, é circular/vazamento de dados. Recomendação para a etapa 2 (ML): tratar `Electricity_MWh`, `Natural_Gas_m3h`, `Steam_Tons_h` e o outro target como candidatos a **excluir** (ou usar com muita justificativa) do conjunto de features de cada modelo, mantendo como preditores principalmente as variáveis controláveis, de estado e externas.

## 1.6 Principais restrições operacionais (a partir dos limites observados nos dados)

| Variável | Faixa observada | Uso como restrição na otimização |
|---|---|---|
| Feedstock_Flow_m3h | 341 – 759 | Limite inferior/superior de vazão de alimentação |
| Reactor_Temp_C | 676 – 966 | Limite de temperatura segura do reator |
| Reactor_Pressure_Bar | 20,7 – 43,6 | Limite de pressão segura do reator |
| Valve_Opening_Percent | 20 – 95 | Abertura mínima/máxima da válvula |
| Sensor_Health_Index | 0,56 – 1,00 | Gatilho de manutenção (quanto menor, pior) |
| Vibration_Level_mm_s | 0,50 – 8,50 | Gatilho de manutenção (quanto maior, pior) |

(Definir os limites finais de negócio deve considerar também literatura/normas do setor, não só o range estatístico do dataset — vale citar isso no relatório.)

## 1.7 Regras de negócio sugeridas (ponto de partida para discussão em grupo)

1. A produção (`Product_Yield_Tons`) não pode ser reduzida abaixo de um piso mínimo contratual/comercial ao otimizar energia.
2. `Reactor_Temp_C` e `Reactor_Pressure_Bar` devem ficar dentro da faixa segura de operação do equipamento (evitar extrapolar o range observado sem justificativa de engenharia).
3. Quando `Sensor_Health_Index` cai abaixo de um limiar (ex.: percentil 25 ≈ 0,73) **e/ou** `Vibration_Level_mm_s` sobe acima de um limiar (ex.: percentil 75 ≈ 6,46), o cenário de manutenção deve ser avaliado.
4. `Catalyst_Type` e `Catalyst_Age_Days` só mudam em eventos de manutenção/recarga — não são ajustáveis em tempo real como as demais variáveis controláveis.

---

Isso cobre o pedido de "Entendimento do negócio" (objetivo, variáveis controláveis/estado/externas, restrições, regras de negócio + tabela). O dicionário de dados completo, com tipo, unidade e estatísticas de cada coluna, está em [`02_dicionario_dados.md`](02_dicionario_dados.md).
