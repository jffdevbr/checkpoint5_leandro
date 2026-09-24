# 2. Dicionário de Dados — petrochemical_advanced_data.csv

**Grão:** 1 linha = 1 leitura operacional de 1 planta, em 1 timestamp (grade de 4 em 4 horas, 2020-01-01 a 2024-07-24, 10.000 linhas, 3 plantas, sem timestamps repetidos — ver [`01_entendimento_negocio.md`](01_entendimento_negocio.md) para a análise de granularidade e independência dos registros).

| # | Coluna | Tipo | Descrição | Unidade | Faixa observada (min – max) | Papel |
|---|---|---|---|---|---|---|
| 1 | `Timestamp` | datetime | Instante da leitura, grade fixa de 4h | — | 2020-01-01 00:00 a 2024-07-24 12:00 | Identificador / temporal |
| 2 | `Unit_Name` | categórica (3 valores) | Planta/unidade industrial que gerou a leitura | — | Ethylene_Plant_01, Ammonia_Unit_02, Methanol_Complex_03 | Identificador / segmentação |
| 3 | `Catalyst_Type` | categórica (3 valores) | Tipo de catalisador em uso no reator | — | Iron_Standard_V5, Cobalt_Premium_Z2, Platinum_Base_X1 | Controlável (decisão de longo prazo) |
| 4 | `Catalyst_Age_Days` | inteiro | Idade do catalisador atualmente em uso | dias | 1 – 364 | Variável de estado |
| 5 | `Sensor_Health_Index` | float | Índice de saúde dos sensores/instrumentação da planta | índice (0–1) | 0,560 – 1,000 | Variável de estado (condição do equipamento) |
| 6 | `Vibration_Level_mm_s` | float | Nível de vibração medido no equipamento | mm/s | 0,50 – 8,50 | Variável de estado (condição do equipamento) |
| 7 | `Valve_Opening_Percent` | float | Abertura da válvula de controle de processo | % | 20,0 – 95,0 | Controlável |
| 8 | `Feedstock_Flow_m3h` | float | Vazão de matéria-prima alimentada ao reator | m³/h | 341,0 – 758,5 | Controlável |
| 9 | `Reactor_Temp_C` | float | Temperatura de operação do reator | °C | 675,8 – 966,4 | Controlável (setpoint) |
| 10 | `Reactor_Pressure_Bar` | float | Pressão de operação do reator | bar | 20,7 – 43,6 | Controlável (setpoint) |
| 11 | `Electricity_MWh` | float | Consumo de eletricidade no período | MWh | 15,0 – 30,0 | Consumo de energia (usar com cautela — ver alerta de leakage) |
| 12 | `Natural_Gas_m3h` | float | Consumo de gás natural | m³/h | 3012,5 – 5180,9 | Consumo de energia (usar com cautela — ver alerta de leakage) |
| 13 | `Steam_Tons_h` | float | Consumo/geração de vapor | t/h | 40,0 – 90,0 | Consumo de energia (usar com cautela — ver alerta de leakage) |
| 14 | `Ambient_Temp_C` | float | Temperatura ambiente no momento da leitura | °C | -10,0 – 52,0 | Variável externa (não controlável) |
| 15 | `Product_Yield_Tons` | float | Produção de produto obtida | toneladas | 40,6 – 134,0 | **Target 1** — identidade exata confirmada, ver alerta de leakage |
| 16 | `Energy_Intensity` | float | Intensidade energética do processo (energia consumida por unidade produzida) | energia/tonelada | 1,33 – 6,39 | **Target 2** — métrica derivada (confirmado, ver alerta de leakage) |

## Qualidade dos dados

- **Valores ausentes:** nenhum (0 em todas as 16 colunas, 10.000 linhas).
- **Duplicatas:** nenhuma linha duplicada; nenhum `Timestamp` repetido.
- **Valores fora do esperado:** `Ambient_Temp_C` negativo em 1.637 linhas (16%) — plausível (estações frias/inverno), não é erro.
- **Balanceamento categórico:** `Unit_Name` e `Catalyst_Type` estão bem balanceados entre suas categorias (~33% cada).

## Correlações relevantes com os targets

**Com `Energy_Intensity`:**
| Variável | Correlação |
|---|---|
| Product_Yield_Tons | -0,819 |
| Sensor_Health_Index | -0,658 |
| Feedstock_Flow_m3h | -0,507 |
| Natural_Gas_m3h | +0,405 |
| Vibration_Level_mm_s | +0,337 |
| Electricity_MWh | +0,326 |

**Com `Product_Yield_Tons`:**
| Variável | Correlação |
|---|---|
| Sensor_Health_Index | +0,779 |
| Feedstock_Flow_m3h | +0,625 |
| Vibration_Level_mm_s | -0,381 |
| Energy_Intensity | -0,819 |

(Demais variáveis têm correlação próxima de zero com ambos os targets.)

## Alerta de target leakage — identidades confirmadas

Regressão sobre os dados reais confirma, com **R² = 1,000000**, que os dois targets não são medições independentes — são calculados a partir de outras colunas do próprio dataset:

```
Energy_Intensity   = (3,6 × Electricity_MWh + 0,035 × Natural_Gas_m3h) / Product_Yield_Tons
Product_Yield_Tons = 0,18 × Feedstock_Flow_m3h × Sensor_Health_Index
```

(`Steam_Tons_h` tem coeficiente ~0 na primeira fórmula — não entra no cálculo, apesar de estar listada como "consumo de energia").

**Duas consequências:**

1. **Leakage de ML:** usar `Electricity_MWh`, `Natural_Gas_m3h` ou `Product_Yield_Tons` como feature para prever `Energy_Intensity` (ou vice-versa) é circular — o R² fica artificialmente alto e o modelo não serve para otimizar uma configuração nova. Essas colunas devem ficar fora do conjunto de features de cada modelo (ver seção 4 do notebook).
2. **Os dois targets são algebricamente acoplados:** o consumo bruto de energia (`Electricity_MWh`, `Natural_Gas_m3h`, `Steam_Tons_h`) não correlaciona com nenhuma variável operacional medida (todas as correlações abaixo de 0,09 em módulo) — comporta-se como ruído estatístico do dataset sintético, não como função da configuração da planta. Por isso, `Energy_Intensity` acaba sendo, na prática, quase só função de `Product_Yield_Tons`: só conhecer o Yield real já explica ~71% da variância de `Energy_Intensity` (usando a média fixa de energia no lugar do valor real), mais do que os modelos de ML treinados com todas as features (~68% de R²). Isso não invalida os dois modelos — cada um responde a uma pergunta de negócio diferente (`Energy_Intensity` → configuração operacional de menor custo; `Product_Yield_Tons` → sinal de necessidade de manutenção, ver seção 1.1 de [`01_entendimento_negocio.md`](01_entendimento_negocio.md)) —, mas explica por que, neste dataset, "reduzir a intensidade energética" se dá sobretudo por "aumentar a produção", e não por uma alavanca de eficiência energética independente.

Investigação completa nas seções 2.1.b–2.1.f e 4 do notebook [`01_pipeline_completo.ipynb`](../notebooks/01_pipeline_completo.ipynb).
