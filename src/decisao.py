"""
Lógica de decisão econômica compartilhada entre o notebook, a API
(`app/main.py`) e `scripts/bootstrap_models.py`.

- `economia`: resultado de uma janela de 4 h, com o custo do feedstock.
- `otimizar_lp`: Rota A da seção 6 do notebook e otimizador da API.
- Manutenção (seção 7): `taxa_falha_dia`/`prob_falha`, `projetar_estado`,
  `avaliar_cenario`, `varrer_postergacao`, `idade_otima_catalisador` e
  `recomendar_manutencao`, com os dois tipos (mecânica e catalisador).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import src.config as cfg
from src.features import montar_X_lote


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


def _prever(modelo: Any, vazoes: Sequence[float], fixos: Sequence[float], ctx: dict[str, Any],
            features: Sequence[str] | None = None) -> np.ndarray:
    """Previsões para várias vazões em UMA chamada (predict linha a linha domina o custo)."""
    return modelo.predict(montar_X_lote([[f, *fixos] for f in vazoes], ctx, features))


def taxa_falha_dia(vibracao: float) -> float:
    """Taxa diária de falha mecânica (hazard) para uma vibração, em 1/dia.

    PREMISSA: P(falha em 30 dias) é logística na vibração, ancorada em
    `cfg.ANCORAS_FALHA_ISO` (zonas da ISO 10816). Idade do catalisador e Health
    não entram: o primeiro é um efeito de energia (H2) e o segundo é eficiência
    do processo, não confiabilidade mecânica.
    """
    (v1, p1), (v2, p2) = cfg.ANCORAS_FALHA_ISO
    logit1, logit2 = np.log(p1 / (1 - p1)), np.log(p2 / (1 - p2))
    escala = (v2 - v1) / (logit2 - logit1)
    p30 = 1 / (1 + np.exp(-(vibracao - v2) / escala - logit2))
    return float(-np.log1p(-p30) / 30)


def prob_falha(ctx: dict[str, Any], horizonte_h: float = cfg.HORIZONTE_DECISAO_H) -> float:
    """P(falha mecânica no horizonte) com a vibração atual constante."""
    return float(-np.expm1(-taxa_falha_dia(ctx["Vibration_Level_mm_s"]) * horizonte_h / 24))


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

    def resultados(fs: Sequence[float]) -> list[dict[str, float]]:
        eis = _prever(modelo_energia, fs, setpoints_fixos, ctx, features)
        yds = _prever(modelo_yield, fs, setpoints_fixos, ctx, features)
        return [economia(float(ei), float(yd), f, preco_feedstock) for f, ei, yd in zip(fs, eis, yds)]

    r_lb, r_ub, r_meio = resultados([lb, ub, (lb + ub) / 2])
    s_y = (r_ub["producao_ton"] - r_lb["producao_ton"]) / (ub - lb)
    s_m = (r_ub["margem"] - r_lb["margem"]) / (ub - lb)
    linear_meio = (r_lb["margem"] + r_ub["margem"]) / 2
    vazao_necessaria = lb + (producao_minima - r_lb["producao_ton"]) / s_y

    res = linprog(c=[-s_m], A_ub=[[-s_y]], b_ub=[r_lb["producao_ton"] - s_y * lb - producao_minima],
                  bounds=[(lb, ub)], method="highs")
    viavel = res.status == 0
    f_otimo = float(res.x[0]) if viavel else ub
    # o ótimo do LP costuma cair num limite: reaproveita a previsão já feita
    out = r_lb if f_otimo == lb else r_ub if f_otimo == ub else resultados([f_otimo])[0]
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


# ================================================================ manutenção
JANELAS_POR_DIA = 24 / cfg.HORAS_POR_JANELA

TIPOS_MANUTENCAO = {
    "mecanica": {"custo": cfg.CUSTO_MANUTENCAO_PROGRAMADA, "parada_h": cfg.DOWNTIME_MANUTENCAO_H},
    "catalisador": {"custo": cfg.CUSTO_TROCA_CATALISADOR, "parada_h": cfg.DOWNTIME_TROCA_CATALISADOR_H},
    # mesma parada: os serviços somam, o tempo parado é o do mais longo
    "combinada": {"custo": cfg.CUSTO_MANUTENCAO_PROGRAMADA + cfg.CUSTO_TROCA_CATALISADOR,
                  "parada_h": max(cfg.DOWNTIME_MANUTENCAO_H, cfg.DOWNTIME_TROCA_CATALISADOR_H)},
}


def degradado(vibracao: float) -> bool:
    return vibracao > cfg.LIMIAR_VIBRACAO_REGIME


@lru_cache(maxsize=16)
def _health_por_regime(modelo_health: Any, regime_degradado: bool) -> float:
    return float(modelo_health.predict(pd.DataFrame({"Vib_Degradado": [float(regime_degradado)]}))[0])


def health_esperado(modelo_health: Any, vibracao: float) -> float:
    """E[Health | regime da vibração] — etapa A do modelo de Yield."""
    return _health_por_regime(modelo_health, degradado(vibracao))


def ciclo_mecanico_dias(taxa_vibracao: float = cfg.TAXA_DEGRADACAO_VIBRACAO) -> float:
    """Dias entre uma manutenção mecânica e o próximo cruzamento do limiar de regime."""
    return (cfg.LIMIAR_VIBRACAO_REGIME - cfg.VIBRACAO_POS_MANUTENCAO) / taxa_vibracao


def aplicar_manutencao(ctx: dict[str, Any], tipo: str, modelo_health: Any) -> dict[str, Any]:
    """Estado logo após a intervenção (o Health é o esperado do regime resultante)."""
    novo = dict(ctx)
    if tipo in ("mecanica", "combinada"):
        novo["Vibration_Level_mm_s"] = cfg.VIBRACAO_POS_MANUTENCAO
    if tipo in ("catalisador", "combinada"):
        novo["Catalyst_Age_Days"] = 0.0
    novo["Sensor_Health_Index"] = health_esperado(modelo_health, novo["Vibration_Level_mm_s"])
    return novo


def projetar_estado(ctx: dict[str, Any], dias: float, modelo_health: Any,
                    taxa_vibracao: float = cfg.TAXA_DEGRADACAO_VIBRACAO) -> dict[str, Any]:
    """Estado esperado `dias` à frente, sem intervenção.

    PREMISSA de dinâmica (os dados são cross-sectional, autocorrelação ≈ 0, então ela
    não vem deles): a vibração cresce `taxa_vibracao` mm/s por dia e a idade, 1 dia/dia.
    O Health de um dia futuro é E[Health | regime]: dentro do regime ele é um sorteio
    uniforme independente da vibração (H1) e sem memória (H7), então a leitura de hoje
    não informa o de amanhã — ela só vale para a operação de hoje.
    """
    vib = ctx["Vibration_Level_mm_s"] + taxa_vibracao * dias
    return {**ctx, "Vibration_Level_mm_s": vib, "Catalyst_Age_Days": ctx["Catalyst_Age_Days"] + dias,
            "Sensor_Health_Index": health_esperado(modelo_health, vib)}


def criar_margem_otima(modelo_yield: Any, modelo_energia: Any, bounds_vazao: tuple[float, float],
                       setpoints_fixos: Sequence[float], producao_minima: float = cfg.PRODUCAO_MINIMA_TON,
                       features: Sequence[str] | None = None) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Operação ótima (LP) para um estado, com cache. Os modelos de energia e yield
    só dependem do estado via Health e idade, que formam a chave do cache."""
    cache: dict[tuple[float, float], dict[str, Any]] = {}

    def margem_otima(estado: dict[str, Any]) -> dict[str, Any]:
        chave = (round(estado["Sensor_Health_Index"], 4), round(estado["Catalyst_Age_Days"] * 2) / 2)
        if chave not in cache:
            cache[chave] = otimizar_lp(modelo_yield, modelo_energia, estado, bounds_vazao, setpoints_fixos,
                                       producao_minima, features=features)
        return cache[chave]

    return margem_otima


def avaliar_cenario(ctx: dict[str, Any], tipo: str | None, dia: int | None, horizonte_dias: int,
                    margem_otima: Callable[[dict[str, Any]], dict[str, Any]], modelo_health: Any,
                    taxa_vibracao: float = cfg.TAXA_DEGRADACAO_VIBRACAO, nome: str | None = None) -> dict[str, Any]:
    """Resultado esperado no horizonte, re-otimizando a operação a cada dia com o estado do dia.

    `tipo` ∈ TIPOS_MANUTENCAO (ou None = sem manutenção), feita no dia `dia`.
    resultado_liquido = margem operacional − margem perdida na parada − custo do serviço
                        − P(falha no horizonte) · custo da falha não programada.
    """
    base, dias_desde = dict(ctx), 0
    margem_total = perda_parada = hazard_acum = 0.0
    producao_total = energia_total = custo_energia_total = 0.0
    dias_inviaveis = 0
    estado = projetar_estado(ctx, 0, modelo_health, taxa_vibracao)
    estado_na_manutencao = None
    for t in range(horizonte_dias):
        if tipo is not None and t == dia:
            estado_na_manutencao = estado
            margem_hora = margem_otima(estado)["margem"] / cfg.HORAS_POR_JANELA
            perda_parada = max(margem_hora, 0.0) * TIPOS_MANUTENCAO[tipo]["parada_h"]
            base, dias_desde = aplicar_manutencao(estado, tipo, modelo_health), 0
        estado = projetar_estado(base, dias_desde, modelo_health, taxa_vibracao)
        r = margem_otima(estado)
        if t == 0:
            operacao_inicial = r
        margem_total += r["margem"] * JANELAS_POR_DIA
        producao_total += r["producao_ton"]
        energia_total += r["energy_intensity"] * r["producao_ton"]
        custo_energia_total += r["custo_energia"]
        dias_inviaveis += not r["viavel"]
        hazard_acum += taxa_falha_dia(estado["Vibration_Level_mm_s"])
        dias_desde += 1

    feito = tipo is not None and dia is not None and dia < horizonte_dias
    custo_servico = TIPOS_MANUTENCAO[tipo]["custo"] if feito else 0.0
    p = float(-np.expm1(-hazard_acum))
    return {
        "cenario": nome or ("S1 — sem manutenção" if not feito else f"{tipo} no dia {dia}"),
        "tipo": tipo if feito else None,
        "dia_manutencao": dia if feito else None,
        "horizonte_dias": horizonte_dias,
        "margem_operacional": margem_total,
        "perda_margem_parada": perda_parada if feito else 0.0,
        "custo_manutencao": custo_servico,
        "prob_falha": p,
        "custo_esperado_falha": p * cfg.CUSTO_FALHA_NAO_PROGRAMADA,
        "resultado_liquido": margem_total - (perda_parada if feito else 0.0) - custo_servico
                             - p * cfg.CUSTO_FALHA_NAO_PROGRAMADA,
        "dias_producao_minima_inviavel": dias_inviaveis,
        # médias por janela de 4 h ao longo do horizonte (a operação muda dia a dia)
        "producao_media_ton": producao_total / horizonte_dias,
        "energy_intensity_media": energia_total / producao_total,
        "custo_energia_medio": custo_energia_total / horizonte_dias,
        "operacao_inicial": operacao_inicial,  # LP do dia 0 (setpoints de hoje neste cenário)
        "estado_na_manutencao": estado_na_manutencao if feito else None,
        "estado_final": estado,
    }


def varrer_postergacao(ctx: dict[str, Any], tipo: str, margem_otima: Callable[[dict[str, Any]], dict[str, Any]],
                       modelo_health: Any, dias_max: int = 60,
                       taxa_vibracao: float = cfg.TAXA_DEGRADACAO_VIBRACAO) -> dict[str, Any]:
    """S3 para N = 0..dias_max (N = 0 é S2), pelo critério de renovação.

    Um horizonte fixo pune quem mantém cedo (vê o próximo cruzamento do limiar) e
    ignora que adiantar consome vida útil. Por isso cada N é avaliado na janela
    [0, N + L], com L = um ciclo pós-manutenção (`ciclo_mecanico_dias`), e comparado
    pelo resultado esperado POR DIA. O custo por dia é a margem diária do estado
    recém-mantido menos esse resultado (inclui parada, serviço, risco e perda de eficiência).

    - n_otimo: N de menor custo por dia entre os que não violam a produção mínima (R5);
    - n_indiferenca: primeiro N depois do ótimo em que adiar custa mais que fazer hoje.
    """
    ciclo = int(round(ciclo_mecanico_dias(taxa_vibracao)))
    margem_ideal_dia = margem_otima(aplicar_manutencao(ctx, tipo, modelo_health))["margem"] * JANELAS_POR_DIA
    curva = []
    for n in range(dias_max + 1):
        janela = n + ciclo
        c = avaliar_cenario(ctx, tipo, n, janela, margem_otima, modelo_health, taxa_vibracao)
        s1 = avaliar_cenario(ctx, None, None, janela, margem_otima, modelo_health, taxa_vibracao)
        c["resultado_por_dia"] = c["resultado_liquido"] / janela
        c["custo_por_dia"] = margem_ideal_dia - c["resultado_por_dia"]
        c["sem_manutencao_por_dia"] = s1["resultado_liquido"] / janela
        curva.append(c)
    custo = np.array([c["custo_por_dia"] for c in curva])
    # R5 é restrição, não custo: só concorrem os N com o mínimo de dias de meta inviável
    inviaveis = np.array([c["dias_producao_minima_inviavel"] for c in curva])
    n_otimo = int(np.where(inviaveis == inviaveis.min(), custo, np.inf).argmin())
    depois = np.nonzero(custo[n_otimo:] > custo[0])[0]
    melhor = curva[n_otimo]
    return {
        "tipo": tipo,
        "ciclo_dias": ciclo,
        "margem_ideal_dia": margem_ideal_dia,
        "curva": curva,
        "n_otimo": n_otimo,
        "n_indiferenca": int(n_otimo + depois[0]) if n_otimo > 0 and depois.size else None,
        "vale_a_pena": bool(melhor["resultado_por_dia"] > melhor["sem_manutencao_por_dia"]),
        "ganho_por_dia_vs_sem_manutencao": float(melhor["resultado_por_dia"] - melhor["sem_manutencao_por_dia"]),
    }


def custo_degradacao_catalisador_dia() -> float:
    """k: R$ por dia, por dia de idade do catalisador (gás extra -> energia extra)."""
    return (cfg.CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE * cfg.COEF_GAS_ENERGIA
            * cfg.GAS_POR_DIA_IDADE * JANELAS_POR_DIA)


def idade_otima_catalisador(margem_por_hora: float, custo_troca: float = cfg.CUSTO_TROCA_CATALISADOR,
                            parada_h: float = cfg.DOWNTIME_TROCA_CATALISADOR_H,
                            parada_ja_paga_h: float = 0.0) -> dict[str, float]:
    """Idade ótima de troca: substituição com custo de operação crescendo linearmente.

    Custo médio por dia num ciclo de T dias = C/T + k·T/2, mínimo em T* = sqrt(2C/k).
    C = serviço + margem perdida nas horas de parada que ainda não estão pagas por
    outra intervenção (`parada_ja_paga_h` > 0 quando a troca pega carona na parada mecânica).
    """
    k = custo_degradacao_catalisador_dia()
    custo_total = custo_troca + max(margem_por_hora, 0.0) * max(parada_h - parada_ja_paga_h, 0.0)
    return {
        "t_otimo_dias": float(np.sqrt(2 * custo_total / k)),
        "custo_total_troca": float(custo_total),
        "k_reais_por_dia_por_dia": float(k),
        "custo_medio_diario_no_otimo": float(np.sqrt(2 * custo_total * k)),
    }


def recomendar_manutencao(ctx: dict[str, Any], margem_otima: Callable[[dict[str, Any]], dict[str, Any]],
                          modelo_health: Any, dias_max: int = 60) -> dict[str, Any]:
    """Decisão de manutenção para o estado atual, separando os dois tipos.

    A leitura atual de Health só decide a R5 (a operação de hoje); todo o resto usa
    E[Health | regime], porque a leitura de hoje não informa as futuras (H1/H7).
    - Mecânica: R5 (produção mínima inviável agora ou no regime => obrigatória já); senão, o N
      ótimo da varredura de postergação (se a intervenção vale a pena no horizonte).
    - Catalisador: trocar se idade >= T* isolado; ou junto com a parada mecânica
      se idade >= T* acoplado (a parada já está paga).
    """
    agora = margem_otima(ctx)                                        # leitura atual: só para a R5
    no_regime = margem_otima(projetar_estado(ctx, 0, modelo_health))   # estado esperado: todo o resto
    margem_hora = no_regime["margem"] / cfg.HORAS_POR_JANELA
    varredura = varrer_postergacao(ctx, "mecanica", margem_otima, modelo_health, dias_max)
    t_isolado = idade_otima_catalisador(margem_hora)
    t_acoplado = idade_otima_catalisador(margem_hora, parada_ja_paga_h=cfg.DOWNTIME_MANUTENCAO_H)
    idade = ctx["Catalyst_Age_Days"]

    if not agora["viavel"] or not no_regime["viavel"]:
        onde = "na leitura atual" if not agora["viavel"] else "no Health esperado do regime atual"
        mecanica = {"fazer": True, "quando_dias": 0, "motivo": f"R5 — produção mínima inviável {onde}: "
                    "manutenção mecânica obrigatória."}
    elif varredura["vale_a_pena"]:
        n = varredura["n_otimo"]
        indif = varredura["n_indiferenca"]
        mecanica = {"fazer": True, "quando_dias": n, "motivo": (
            "Imediata: adiar só aumenta o custo esperado por dia." if n == 0 else
            f"Programar em {n} dias (menor custo esperado por dia)"
            + (f"; adiar além de {indif} dias fica pior que fazer hoje." if indif is not None else "."))}
    else:
        mecanica = {"fazer": False, "quando_dias": None,
                    "motivo": "Não se paga: o risco e a perda de eficiência evitados valem menos que a parada."}

    parada_mecanica_proxima = mecanica["fazer"] and mecanica["quando_dias"] <= 30
    if idade >= t_isolado["t_otimo_dias"]:
        catalisador = {"trocar": True, "junto_da_mecanica": parada_mecanica_proxima,
                       "motivo": f"R6 — idade {idade:.0f} d >= T* isolado de {t_isolado['t_otimo_dias']:.0f} d."}
    elif parada_mecanica_proxima and idade >= t_acoplado["t_otimo_dias"]:
        catalisador = {"trocar": True, "junto_da_mecanica": True,
                       "motivo": f"R6 — aproveitar a parada mecânica: idade {idade:.0f} d >= T* acoplado de "
                                 f"{t_acoplado['t_otimo_dias']:.0f} d."}
    else:
        limites = f"{t_isolado['t_otimo_dias']:.0f} d isolado"
        if parada_mecanica_proxima:
            limites += f", {t_acoplado['t_otimo_dias']:.0f} d acoplado"
        catalisador = {"trocar": False, "junto_da_mecanica": False,
                       "motivo": f"Idade {idade:.0f} d abaixo de T* ({limites})."}
    return {"mecanica": mecanica, "catalisador": catalisador, "varredura_mecanica": varredura,
            "t_otimo_catalisador": {"isolado": t_isolado, "acoplado_a_mecanica": t_acoplado},
            "operacao_atual_viavel": agora["viavel"], "regime_atual_viavel": no_regime["viavel"]}
