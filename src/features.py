"""
Feature engineering compartilhado entre o notebook, a API (`app/main.py`) e
`scripts/bootstrap_models.py`.

Conjunto de features atual (`build_features`, `FEATURES_*`) é o BASELINE
herdado das três cópias duplicadas que existiam antes da Etapa 0. A Etapa 2
do plano revisa este conjunto (remove `Delta_Temp`/`Severidade`/
`Carga_por_Abertura`/`Idade_Norm`, separa features por target) — até lá,
`FEATURES_ENERGIA` e `FEATURES_YIELD` são intencionalmente iguais.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

import src.config as cfg


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva as features de engenharia a partir do dataframe bruto."""
    X = df.copy()
    X["Delta_Temp"] = X["Reactor_Temp_C"] - X["Ambient_Temp_C"]
    X["Severidade"] = X["Reactor_Temp_C"] * X["Reactor_Pressure_Bar"]
    X["Carga_por_Abertura"] = X["Feedstock_Flow_m3h"] / X["Valve_Opening_Percent"].replace(0, np.nan)
    X["Idade_Norm"] = X["Catalyst_Age_Days"] / cfg.LIMITE_IDADE_CATALISADOR
    return X


# Baseline herdado — revisado na Etapa 2 (feature engineering causal).
FEATURES_ESTADO = list(cfg.ESTADO)
FEATURES_ENERGIA = (
    cfg.CONTROLAVEIS
    + cfg.ESTADO
    + ["Ambient_Temp_C", "Delta_Temp", "Severidade", "Carga_por_Abertura", "Idade_Norm"]
)
FEATURES_YIELD = list(FEATURES_ENERGIA)


def montar_X(x: Sequence[float], ctx: dict[str, Any], features: Sequence[str] | None = None) -> pd.DataFrame:
    """Monta a linha de features a partir de (decisão x, contexto ctx).

    `x` segue a ordem de `cfg.CONTROLAVEIS`; `ctx` é um dict com o estado do
    equipamento e as condições externas (ex.: saída de `Contexto.model_dump()`
    na API, ou o dict `CTX` no notebook).
    """
    d: dict[str, Any] = dict(zip(cfg.CONTROLAVEIS, x))
    d.update(ctx)
    d["Delta_Temp"] = d["Reactor_Temp_C"] - d["Ambient_Temp_C"]
    d["Severidade"] = d["Reactor_Temp_C"] * d["Reactor_Pressure_Bar"]
    d["Carga_por_Abertura"] = d["Feedstock_Flow_m3h"] / d["Valve_Opening_Percent"]
    d["Idade_Norm"] = d["Catalyst_Age_Days"] / cfg.LIMITE_IDADE_CATALISADOR
    cols = list(features) if features is not None else FEATURES_ENERGIA
    return pd.DataFrame([d])[cols]


def limites_operacionais(df: pd.DataFrame, cols: Sequence[str], lo: float = 5, hi: float = 95) -> dict[str, tuple[float, float]]:
    """Bounds operacionais por percentil (5/95 por padrão) das colunas dadas."""
    return {c: (float(np.percentile(df[c], lo)), float(np.percentile(df[c], hi))) for c in cols}
