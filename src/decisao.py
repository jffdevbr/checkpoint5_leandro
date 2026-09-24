"""
Lógica de decisão econômica compartilhada entre o notebook, a API
(`app/main.py`) e `scripts/bootstrap_models.py`.

`economia` inclui o custo do feedstock (premissa `PRECO_FEEDSTOCK_M3`).
`otimizar_lp` é a Rota A da seção 6 do notebook e o otimizador da API.

`projetar_estado`, `avaliar_cenario` e `idade_otima_catalisador` são stubs —
a lógica é definida na Etapa 5 do plano (cenários de manutenção).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog

import src.config as cfg
from src.features import montar_X


def economia(ei: float, yd: float, vazao: float, preco_feedstock: float = cfg.PRECO_FEEDSTOCK_M3) -> dict[str, float]:
    """Resultado econômico de UMA janela de operação (4 h), em R$."""
    custo_energia = cfg.CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE * ei * yd
    custo_feedstock = preco_feedstock * cfg.HORAS_POR_JANELA * vazao
    receita = cfg.PRECO_PRODUTO_TON * yd
    return {
        "energy_intensity": ei,
        "producao_ton": yd,
        "custo_energia": custo_energia,
        "custo_feedstock": custo_feedstock,
        "receita": receita,
        "margem": receita - custo_energia - custo_feedstock,
    }


def _prever(modelo: Any, vazao: float, fixos: Sequence[float], ctx: dict[str, Any],
            features: Sequence[str] | None = None) -> float:
    return float(modelo.predict(montar_X([vazao, *fixos], ctx, features))[0])


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


def otimizar_lp(
    modelo_yield: Any,
    modelo_energia: Any,
    ctx: dict[str, Any],
    bounds_vazao: tuple[float, float],
    setpoints_fixos: Sequence[float],
    producao_minima: float = cfg.PRODUCAO_MINIMA_TON,
    preco_feedstock: float = cfg.PRECO_FEEDSTOCK_M3,
    features: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rota A — LP com `linprog` na vazão, com T/P/V fixos e o estado em `ctx`.

    No log-log, Y = k·H·F (elasticidade 1) e a energia total EI·Y quase não
    depende de F (elasticidade de EI ≈ −1), então a margem da janela é linear
    em F:  max (p·k·H − c_E' − 4·c_f)·F  s.a.  k·H·F ≥ Y_min,  F ∈ [lb, ub].
    Os coeficientes saem dos próprios modelos, por secante entre lb e ub
    (exata nos limites); `erro_linearizacao_pct` mede o desvio no meio.
    Se Y_min for inviável, opera em F = ub (produção máxima) e sinaliza.
    """
    lb, ub = bounds_vazao

    def resultado(f: float) -> dict[str, float]:
        return economia(_prever(modelo_energia, f, setpoints_fixos, ctx, features),
                        _prever(modelo_yield, f, setpoints_fixos, ctx, features), f, preco_feedstock)

    r_lb, r_ub, r_meio = resultado(lb), resultado(ub), resultado((lb + ub) / 2)
    s_y = (r_ub["producao_ton"] - r_lb["producao_ton"]) / (ub - lb)
    s_m = (r_ub["margem"] - r_lb["margem"]) / (ub - lb)
    linear_meio = (r_lb["margem"] + r_ub["margem"]) / 2
    vazao_necessaria = lb + (producao_minima - r_lb["producao_ton"]) / s_y

    res = linprog(c=[-s_m], A_ub=[[-s_y]], b_ub=[r_lb["producao_ton"] - s_y * lb - producao_minima],
                  bounds=[(lb, ub)], method="highs")
    viavel = res.status == 0
    f_otimo = float(res.x[0]) if viavel else ub
    out = resultado(f_otimo)
    return {
        "rota": "A — LP (linprog/HiGHS)",
        "sucesso": viavel,
        "viavel": viavel,
        "mensagem": "Ótimo encontrado." if viavel else (
            f"Produção mínima de {producao_minima:.1f} t inviável: exige vazão de {vazao_necessaria:.1f} m³/h, "
            f"acima do limite operacional de {ub:.1f}. Operando no limite; manutenção obrigatória."),
        "setpoints": dict(zip(cfg.CONTROLAVEIS, [f_otimo, *map(float, setpoints_fixos)])),
        **out,
        "restricao_producao_atendida": out["producao_ton"] >= producao_minima - 1e-6,
        "vazao_necessaria_producao_minima": vazao_necessaria,
        # Variação da margem da janela (R$) por +1 m³/h de capacidade e por +1 t de produção
        # mínima exigida (≤ 0). Conferidos contra diferenças finitas na seção 6.
        "preco_sombra_capacidade": -float(res.upper.marginals[0]) + 0.0 if viavel else None,
        "preco_sombra_producao_minima": float(res.ineqlin.marginals[0]) + 0.0 if viavel else None,
        "lp": {"margem_por_m3h": s_m, "producao_por_m3h": s_y,
               "erro_linearizacao_pct": (linear_meio / r_meio["margem"] - 1) * 100},
    }


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
