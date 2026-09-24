"""
Feature engineering compartilhado entre o notebook, a API (`app/main.py`) e
`scripts/bootstrap_models.py`.

Conjuntos por modelo (definidos na seção 3 do notebook, com ablação na CV):

- Energia (P1), target log(Energy_Intensity): FEATURES_ENERGIA
- Yield (P2), target log(Product_Yield_Tons), em duas etapas:
    A) Health ~ FEATURES_ESTADO             (estado mecânico -> eficiência)
    B) log(Yield) ~ FEATURES_YIELD
- Visão alternativa "Yield sem Health": FEATURES_YIELD_SEM_HEALTH

FEATURES_BASELINE é o conjunto antigo de 12 features, mantido só como linha
de comparação e enquanto os modelos da API forem os do bootstrap.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

import src.config as cfg


def _derivar(d: Any) -> Any:
    """Colunas derivadas; `d` pode ser DataFrame ou dict de escalares."""
    d["log_Flow"] = np.log(d["Feedstock_Flow_m3h"])
    d["log_Health"] = np.log(d["Sensor_Health_Index"])
    d["Vib_Degradado"] = (d["Vibration_Level_mm_s"] > cfg.LIMIAR_VIBRACAO_REGIME) * 1.0
    # Features do baseline antigo — sem papel causal (ver ablação da seção 3).
    d["Delta_Temp"] = d["Reactor_Temp_C"] - d["Ambient_Temp_C"]
    d["Severidade"] = d["Reactor_Temp_C"] * d["Reactor_Pressure_Bar"]
    d["Carga_por_Abertura"] = d["Feedstock_Flow_m3h"] / d["Valve_Opening_Percent"]
    d["Idade_Norm"] = d["Catalyst_Age_Days"] / cfg.LIMITE_IDADE_CATALISADOR
    return d


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features de engenharia a partir do dataframe bruto."""
    return _derivar(df.copy())


FEATURES_ENERGIA = ["log_Flow", "log_Health", "Catalyst_Age_Days"]
FEATURES_YIELD = ["log_Flow", "log_Health"]
FEATURES_ESTADO = ["Vib_Degradado"]
FEATURES_YIELD_SEM_HEALTH = ["log_Flow", "Vib_Degradado"]

FEATURES_BASELINE = (
    cfg.CONTROLAVEIS
    + cfg.ESTADO
    + ["Ambient_Temp_C", "Delta_Temp", "Severidade", "Carga_por_Abertura", "Idade_Norm"]
)

# Catálogo usado na tabela da seção 4.3 e no export (metadata.json).
CATALOGO_FEATURES = [
    {"feature": "log_Flow", "modelo": "Energia, Yield", "tipo": "decisão",
     "justificativa": "Única alavanca operacional; no log-log o coeficiente é a elasticidade direta (≈ −1 em EI, +1 em Yield)."},
    {"feature": "log_Health", "modelo": "Energia, Yield", "tipo": "estado",
     "justificativa": "Fator de eficiência que multiplica a produção; confundidor que precisa ser controlado para isolar o efeito da vazão."},
    {"feature": "Catalyst_Age_Days", "modelo": "Energia", "tipo": "estado",
     "justificativa": "Canal causal via gás (EDA H2: Gas = U(3000,5000) + 0,5·idade); base da decisão de troca de catalisador."},
    {"feature": "Vib_Degradado", "modelo": "Health (etapa A), Yield sem Health", "tipo": "estado",
     "justificativa": "Vibração > 6,5 mm/s muda o regime do Health (EDA H1); é o sinal que indica manutenção mecânica."},
]

EXCLUIDAS = {
    "Electricity_MWh": "Leakage: consumo medido, compõe o numerador de Energy_Intensity.",
    "Natural_Gas_m3h": "Leakage: consumo medido, compõe o numerador de Energy_Intensity.",
    "Steam_Tons_h": "Leakage (consumo medido) e fora da fórmula de Energy_Intensity.",
    "Product_Yield_Tons / Energy_Intensity": "O target de um modelo nunca entra no outro (denominador de EI).",
    "Delta_Temp": "Combina T e Ambient, ambos sem efeito (H4/H5); colinear exato (VIF infinito).",
    "Severidade": "Combina T e P, ambos sem efeito (H5).",
    "Carga_por_Abertura": "Sem ganho na CV e cria alavanca espúria na válvula dentro do otimizador.",
    "Idade_Norm": "Redundante: corr = 1 com Catalyst_Age_Days.",
    "Vibration_Level_mm_s (no modelo de energia)": "Efeito totalmente mediado pelo Health; dentro de cada regime, corr ≈ 0.",
    "Reactor_Temp_C / Reactor_Pressure_Bar / Valve_Opening_Percent": "Sem efeito linear nem não linear (H5); ficam fixos na otimização.",
    "Ambient_Temp_C, Mês, Hora": "Sem sazonalidade e sem efeito (H4).",
    "One-hot Unit_Name / Catalyst_Type": "Plantas e catalisadores não são populações distintas (H6); interação com idade não significativa (H3).",
}


def montar_X(x: Sequence[float], ctx: dict[str, Any], features: Sequence[str] | None = None) -> pd.DataFrame:
    """Monta a linha de features a partir de (decisão x, contexto ctx).

    `x` segue a ordem de `cfg.CONTROLAVEIS`; `ctx` é um dict com o estado do
    equipamento e as condições externas (ex.: saída de `Contexto.model_dump()`
    na API, ou o dict `CTX` no notebook).
    """
    d: dict[str, Any] = dict(zip(cfg.CONTROLAVEIS, x))
    d.update(ctx)
    _derivar(d)
    cols = list(features) if features is not None else FEATURES_ENERGIA
    return pd.DataFrame([d])[cols]


def limites_operacionais(df: pd.DataFrame, cols: Sequence[str], lo: float = 5, hi: float = 95) -> dict[str, tuple[float, float]]:
    """Bounds operacionais por percentil (5/95 por padrão) das colunas dadas."""
    return {c: (float(np.percentile(df[c], lo)), float(np.percentile(df[c], hi))) for c in cols}
