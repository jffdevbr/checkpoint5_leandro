"""
Parâmetros centrais do projeto: caminhos, classificação de variáveis e
premissas econômicas.

IMPORTANTE: os preços e custos abaixo NÃO vêm do dataset — são premissas
assumidas pelo grupo. Toda conclusão econômica do trabalho depende deles,
então eles precisam aparecer explicitamente no relatório e na apresentação,
e devem ser testados em análise de sensibilidade (seção 7 do notebook).
"""

from pathlib import Path

# ---------------------------------------------------------------- caminhos
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw" / "petrochemical_advanced_data.csv"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ------------------------------------------------- classificação de variáveis
# Entrega 1 do enunciado. Esta é a "fonte da verdade" usada pelo notebook,
# pelo otimizador e pela API — mantenha sincronizado com a tabela do relatório.

TARGETS = ["Energy_Intensity", "Product_Yield_Tons"]

# O que o operador consegue ajustar no painel (variáveis de decisão da otimização)
CONTROLAVEIS = [
    "Feedstock_Flow_m3h",
    "Reactor_Temp_C",
    "Reactor_Pressure_Bar",
    "Valve_Opening_Percent",
]

# Estado do ativo: muda devagar, não é escolhido no instante, mas é afetado
# pela decisão de manutenção (é isso que torna manutenção uma alavanca).
ESTADO = [
    "Catalyst_Age_Days",
    "Sensor_Health_Index",
    "Vibration_Level_mm_s",
]

# Fora do controle da operação
EXTERNAS = [
    "Ambient_Temp_C",
    "Unit_Name",
    "Catalyst_Type",
]

# ATENÇÃO — LEAKAGE. Estas são MEDIÇÕES DE CONSUMO, ou seja, consequência da
# operação e provável origem aritmética de Energy_Intensity. Usá-las como
# feature gera R² altíssimo e um modelo inútil para otimizar (para saber a
# energia você precisaria já saber a energia). Confirme a relação na seção 4
# do notebook e mantenha fora do conjunto de features do pipeline.
CONSUMO_ENERGIA = [
    "Electricity_MWh",
    "Natural_Gas_m3h",
    "Steam_Tons_h",
]

# ------------------------------------------------- premissas econômicas (R$)
PRECO_PRODUTO_TON = 2_800.0        # receita por tonelada de produto
PRECO_ELETRICIDADE_MWH = 350.0     # R$/MWh
PRECO_GAS_M3 = 2.50                # R$/m³
PRECO_VAPOR_TON = 80.0             # R$/ton de vapor

# Feedstock: sem ele a margem é linear crescente na vazão com coeficiente
# p·0,18·H (todo m³ vira receita). CALIBRADO para o feedstock ser 70% da
# receita média (faixa típica de petroquímica): 70% × 224.609 / (4 h × 550,7 m³/h).
PRECO_FEEDSTOCK_M3 = 71.4          # R$/m³
HORAS_POR_JANELA = 4               # cada leitura do dataset cobre 4 h

# Energia agregada. CALIBRADO nos dados: verificou-se que
#   Energy_Intensity == (3.6*Electricity_MWh + 0.035*Natural_Gas_m3h) / Product_Yield_Tons
# com R² = 1.000000 (identidade exata — ver seção 4 do notebook).
# Com os preços acima, a razão custo_energia / (Energy_Intensity * Yield)
# fica em 80.74 ± 1.43 (p5–p95 ≈ 78.5–83.2), ou seja, quase constante.
# Logo este fator converte "intensidade × produção" em R$ com erro < 2%.
CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE = 80.74

# ------------------------------------------------- premissas de manutenção
CUSTO_MANUTENCAO_PROGRAMADA = 120_000.0   # parada planejada
CUSTO_FALHA_NAO_PROGRAMADA = 800_000.0    # quebra em operação (parada + danos)
DOWNTIME_MANUTENCAO_H = 24                # horas fora de operação
HORIZONTE_DECISAO_H = 720                 # 30 dias — janela de avaliação

# Manutenção mecânica (PREMISSAS): a vibração volta para a fronteira das zonas
# A/B da ISO 10816 e cresce linearmente até a próxima intervenção. Os dados são
# cross-sectional (autocorrelação ≈ 0), então a taxa NÃO é estimável deles.
VIBRACAO_POS_MANUTENCAO = 2.5             # mm/s (P25 dos dados)
TAXA_DEGRADACAO_VIBRACAO = 0.05           # mm/s por dia

# Risco de falha mecânica em 30 dias, logística na vibração ancorada nas zonas
# da ISO 10816: 1% na fronteira B/C (4,5 mm/s) e 30% na fronteira C/D (7,1).
# PREMISSA — calibrar com o histórico real de falhas antes de usar em produção.
ANCORAS_FALHA_ISO = ((4.5, 0.01), (7.1, 0.30))

# Troca de catalisador (PREMISSAS). O ganho de um catalisador novo é só energia:
# Gás = U(3000, 5000) + GAS_POR_DIA_IDADE·idade (EDA H2, derivado dos dados).
CUSTO_TROCA_CATALISADOR = 120_000.0       # carga + serviço
DOWNTIME_TROCA_CATALISADOR_H = 12         # parada própria; cabe dentro da parada mecânica
GAS_POR_DIA_IDADE = 0.5                   # m³/h de gás por dia de idade (H2)
COEF_GAS_ENERGIA = 0.035                  # peso do gás em Energy_Intensity (seção 2.1.c)

# Gatilhos de condição do equipamento (regras de negócio — entrega 1)
LIMITE_VIBRACAO_ALERTA = 4.5      # mm/s
LIMITE_VIBRACAO_CRITICO = 6.0     # mm/s
LIMITE_SENSOR_HEALTH = 0.70       # abaixo disso, leitura pouco confiável
LIMITE_IDADE_CATALISADOR = 400    # dias

# Limiar de regime do equipamento, DERIVADO DOS DADOS (EDA H1): com vibração
# acima dele o Sensor_Health_Index cai de U(0,70–1,00) para U(0,56–0,80).
# Contaminação cruzada entre regimes = 0 exatamente em 6,5 mm/s.
LIMIAR_VIBRACAO_REGIME = 6.5      # mm/s

# Limites operacionais das variáveis de decisão.
# Estratégia recomendada: NÃO inventar — derivar dos percentis 5/95 dos dados
# históricos (ver src/features.py::limites_operacionais) para que o otimizador
# fique dentro da região onde o modelo de ML foi treinado. Estes valores são
# só o fallback.
# (valores abaixo = percentis 5/95 reais do dataset)
BOUNDS_FALLBACK = {
    "Feedstock_Flow_m3h": (451.92, 648.81),
    "Reactor_Temp_C": (753.12, 886.66),
    "Reactor_Pressure_Bar": (27.11, 36.94),
    "Valve_Opening_Percent": (24.10, 91.34),
}

# Produção mínima que a decisão precisa respeitar (restrição do enunciado:
# "reduzir intensidade energética MANTENDO a produção").
# 80.2 ton é a MÉDIA histórica — "manter a produção" = não ficar abaixo do que
# a planta já entrega hoje. Justifique esta escolha no relatório e teste
# sensibilidade (ex.: 75 / 80 / 85) para mostrar como a fronteira se move.
PRODUCAO_MINIMA_TON = 80.0

RANDOM_STATE = 42
