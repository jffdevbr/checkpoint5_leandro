"""
Lógica de decisão econômica compartilhada entre o notebook, a API
(`app/main.py`) e `scripts/bootstrap_models.py`.

`economia`, `prob_falha` e `gate_automacao` são o código já existente
(antes duplicado entre `app/main.py` e os esqueletos do notebook), só
migrado para dict/parâmetros explícitos em vez de depender do modelo
pydantic `Contexto` da API.

`otimizar_lp`, `projetar_estado`, `avaliar_cenario` e
`idade_otima_catalisador` são stubs — a lógica é definida nas Etapas 4 e 5
do plano (otimização e cenários de manutenção) e ainda não existe em
nenhuma das cópias atuais.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import src.config as cfg


def economia(ei: float, yd: float) -> dict[str, float]:
    """Converte (Energy_Intensity, Yield) em custo de energia, receita e margem."""
    custo_energia = cfg.CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE * ei * yd
    receita = cfg.PRECO_PRODUTO_TON * yd
    return {
        "energy_intensity": ei,
        "producao_ton": yd,
        "custo_energia": custo_energia,
        "receita": receita,
        "margem": receita - custo_energia,
    }


def prob_falha(ctx: dict[str, Any], horizonte_h: float = cfg.HORIZONTE_DECISAO_H) -> float:
    """Risco de falha no horizonte dado o estado do equipamento em `ctx`.

    Forma funcional é PREMISSA (ver Etapa 5 do plano) — calibrar/justificar
    com histórico real de falhas antes de qualquer uso em produção.
    """
    z = (
        -6.0
        + 0.9 * ctx["Vibration_Level_mm_s"]
        + 0.004 * ctx["Catalyst_Age_Days"]
        + 2.0 * (1 - ctx["Sensor_Health_Index"])
    )
    p = 1 / (1 + np.exp(-z))
    return float(np.clip(p * (horizonte_h / cfg.HORIZONTE_DECISAO_H), 0, 1))


def gate_automacao(
    ctx: dict[str, Any],
    setpoints: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """Decide o nível de automação permitido neste ponto de operação.

    A tese do trabalho: o nível de automação não é fixo, é função da
    confiança do modelo no ponto específico em que a decisão está sendo
    tomada.
    """
    motivos: list[str] = []
    if ctx["Sensor_Health_Index"] < cfg.LIMITE_SENSOR_HEALTH:
        motivos.append(
            f"Sensor_Health_Index = {ctx['Sensor_Health_Index']:.2f} < {cfg.LIMITE_SENSOR_HEALTH} "
            "— entrada do modelo não é confiável."
        )
    if ctx["Vibration_Level_mm_s"] > cfg.LIMITE_VIBRACAO_CRITICO:
        motivos.append(
            f"Vibração {ctx['Vibration_Level_mm_s']:.2f} mm/s acima do limite crítico "
            f"({cfg.LIMITE_VIBRACAO_CRITICO}) — regra determinística de segurança."
        )
    for c, v in setpoints.items():
        lb, ub = bounds[c]
        if not (lb - 1e-9 <= v <= ub + 1e-9):
            motivos.append(f"{c} fora da faixa de treino — o modelo estaria extrapolando.")

    if motivos:
        nivel, desc = "L0", "Somente relatório — humano decide."
    elif ctx["Sensor_Health_Index"] > 0.9 and ctx["Vibration_Level_mm_s"] < cfg.LIMITE_VIBRACAO_ALERTA:
        nivel, desc = "L2", "Setpoints ajustados automaticamente; manutenção exige aprovação."
    else:
        nivel, desc = "L1", "Sistema recomenda; humano aprova."
    return {"nivel": nivel, "descricao": desc, "motivos": motivos, "automatizavel": not motivos}


def otimizar_lp(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Rota A — LP com `scipy.optimize.linprog` (Etapa 4 do plano).

    Com o modelo log-log, Y é linear em F para um Health fixo e a energia
    não depende de F, então a margem é linear em F:
        max (p·k·H − c_f)·F  sujeito a  k·H·F ≥ Y_min, F ∈ [lb, ub]
    """
    raise NotImplementedError("Rota A (LP) — implementar na Etapa 4 (Otimização).")


def projetar_estado(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Projeta o estado do equipamento (Health/vibração/idade) no tempo.

    É premissa (dados são cross-sectional, autocorrelação ≈ 0) — ver Etapa 5.
    """
    raise NotImplementedError("Projeção de estado — implementar na Etapa 5 (Cenários de manutenção).")


def avaliar_cenario(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Re-otimiza a operação dado o estado de um cenário de manutenção (Etapa 5)."""
    raise NotImplementedError("Avaliação de cenário — implementar na Etapa 5 (Cenários de manutenção).")


def idade_otima_catalisador(*args: Any, **kwargs: Any) -> float:
    """T* = sqrt(2·C_troca / k_dia) — modelo clássico de substituição com degradação linear (Etapa 5)."""
    raise NotImplementedError("Idade ótima de troca — implementar na Etapa 5 (Cenários de manutenção).")
