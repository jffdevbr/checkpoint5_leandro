"""
API + páginas de apresentação do pipeline.

Rodar:
    .venv\\Scripts\\activate
    uvicorn app.main:app --reload

Depois abra http://127.0.0.1:8000

Os modelos são carregados de models/*.pkl — gerados pela seção 12 do notebook.
Enquanto eles não existirem, a API responde 503 com instrução clara em vez de
quebrar, para você conseguir desenvolver o front antes de treinar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.config as cfg  # noqa: E402
import src.decisao as decisao  # noqa: E402
import src.features as features  # noqa: E402

STATIC = Path(__file__).parent / "static"

app = FastAPI(
    title="Pipeline Petroquímico — ML, Otimização e Decisão",
    description="Endpoint dos modelos + páginas de decisão operacional e de manutenção.",
    version="0.1.0",
)


# ============================================================== carga do modelo
class Artefatos:
    """Carga preguiçosa dos modelos, para a API subir mesmo sem eles."""

    def __init__(self) -> None:
        self.energy = None
        self.yield_ = None
        self.meta: dict[str, Any] = {}
        self.carregado = False

    def carregar(self) -> None:
        if self.carregado:
            return
        import joblib

        pe = cfg.MODELS_DIR / "model_energy.pkl"
        py = cfg.MODELS_DIR / "model_yield.pkl"
        pm = cfg.MODELS_DIR / "metadata.json"
        if not (pe.exists() and py.exists() and pm.exists()):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelos ainda não treinados. Rode a seção 12 do notebook "
                    "notebooks/01_pipeline_completo.ipynb para gerar "
                    "models/model_energy.pkl, models/model_yield.pkl e models/metadata.json."
                ),
            )
        self.energy = joblib.load(pe)
        self.yield_ = joblib.load(py)
        self.meta = json.loads(pm.read_text(encoding="utf-8"))
        self.carregado = True


ART = Artefatos()


# ==================================================================== schemas
class Contexto(BaseModel):
    """Estado do equipamento + condições externas no momento da decisão."""

    Catalyst_Age_Days: float = Field(180, ge=0, le=2000)
    Sensor_Health_Index: float = Field(0.85, gt=0, le=1)
    Vibration_Level_mm_s: float = Field(4.5, ge=0, le=20)
    Ambient_Temp_C: float = Field(20.0, ge=-40, le=60)


class Setpoints(BaseModel):
    Feedstock_Flow_m3h: float = Field(gt=0)
    Reactor_Temp_C: float
    Reactor_Pressure_Bar: float
    Valve_Opening_Percent: float

    def vetor(self) -> list[float]:
        return [getattr(self, c) for c in cfg.CONTROLAVEIS]


class PredictIn(BaseModel):
    setpoints: Setpoints
    contexto: Contexto = Contexto()


class OptimizeIn(BaseModel):
    contexto: Contexto = Contexto()
    producao_minima: float = cfg.PRODUCAO_MINIMA_TON


class MaintenanceIn(BaseModel):
    contexto: Contexto = Contexto()
    dias_postergacao: int = Field(30, ge=0, le=180)
    producao_minima: float = cfg.PRODUCAO_MINIMA_TON


# =================================================================== núcleo
def montar_X(x: list[float], ctx: Contexto) -> pd.DataFrame:
    """Reproduz EXATAMENTE o feature engineering do notebook (seção 3)."""
    feats = ART.meta.get("features") or features.FEATURES_BASELINE
    return features.montar_X(x, ctx.model_dump(), feats)


def prever(x: list[float], ctx: Contexto) -> tuple[float, float]:
    X = montar_X(x, ctx)
    return float(ART.energy.predict(X)[0]), float(ART.yield_.predict(X)[0])


def economia(ei: float, yd: float) -> dict[str, float]:
    return decisao.economia(ei, yd)


def prob_falha(ctx: Contexto, horizonte_h: float = cfg.HORIZONTE_DECISAO_H) -> float:
    return decisao.prob_falha(ctx.model_dump(), horizonte_h)


def bounds() -> dict[str, tuple[float, float]]:
    b = ART.meta.get("bounds") or cfg.BOUNDS_FALLBACK
    return {k: (float(v[0]), float(v[1])) for k, v in b.items()}


def otimizar(ctx: Contexto, producao_minima: float) -> dict[str, Any]:
    B = bounds()

    def objetivo(x: np.ndarray) -> float:
        ei, yd = prever(list(x), ctx)
        return cfg.CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE * ei * yd - cfg.PRECO_PRODUTO_TON * yd

    def restr(x: np.ndarray) -> float:
        _, yd = prever(list(x), ctx)
        return yd - producao_minima

    x0 = np.array([np.mean(B[c]) for c in cfg.CONTROLAVEIS])
    res = minimize(
        objetivo,
        x0,
        method="SLSQP",
        bounds=[B[c] for c in cfg.CONTROLAVEIS],
        constraints=[{"type": "ineq", "fun": restr}],
        options={"maxiter": 300, "ftol": 1e-6},
    )
    ei, yd = prever(list(res.x), ctx)
    return {
        "sucesso": bool(res.success),
        "mensagem": str(res.message),
        "setpoints": {c: float(v) for c, v in zip(cfg.CONTROLAVEIS, res.x)},
        **economia(ei, yd),
        "restricao_producao_atendida": bool(yd >= producao_minima - 1e-6),
    }


def gate_automacao(ctx: Contexto, setpoints: dict[str, float]) -> dict[str, Any]:
    """Decide o nível de automação permitido NESTE ponto de operação.

    A tese do trabalho: o nível de automação não é fixo, é função da confiança
    do modelo no ponto específico em que a decisão está sendo tomada.
    """
    return decisao.gate_automacao(ctx.model_dump(), setpoints, bounds())


# ==================================================================== rotas
@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "modelos_treinados": (cfg.MODELS_DIR / "model_energy.pkl").exists(),
    }


@app.get("/api/metadata")
def metadata() -> dict[str, Any]:
    ART.carregar()
    return {
        "variaveis": {
            "controlaveis": cfg.CONTROLAVEIS,
            "estado": cfg.ESTADO,
            "externas": cfg.EXTERNAS,
            "excluidas_por_leakage": cfg.CONSUMO_ENERGIA,
            "targets": cfg.TARGETS,
        },
        "bounds": bounds(),
        "premissas": ART.meta.get("premissas", {}),
        "metricas_modelos": ART.meta.get("metricas", []),
    }


@app.post("/api/predict")
def predict(body: PredictIn) -> dict[str, Any]:
    """Endpoint do modelo: setpoints + contexto -> energia, produção, custo."""
    ART.carregar()
    ei, yd = prever(body.setpoints.vetor(), body.contexto)
    return economia(ei, yd)


@app.post("/api/optimize")
def optimize(body: OptimizeIn) -> dict[str, Any]:
    """Configuração operacional de menor custo que mantém a produção."""
    ART.carregar()
    out = otimizar(body.contexto, body.producao_minima)
    out["automacao"] = gate_automacao(body.contexto, out["setpoints"])
    return out


@app.post("/api/maintenance")
def maintenance(body: MaintenanceIn) -> dict[str, Any]:
    """Compara os 3 cenários e devolve a recomendação de manutenção."""
    ART.carregar()
    ctx = body.contexto
    N = body.dias_postergacao

    # S1 — sem manutenção: estado atual segue degradando
    ctx_s1 = ctx.model_copy(
        update={
            "Catalyst_Age_Days": ctx.Catalyst_Age_Days + cfg.HORIZONTE_DECISAO_H / 24,
            "Vibration_Level_mm_s": min(ctx.Vibration_Level_mm_s * 1.05, 20),
        }
    )
    # S2 — manutenção imediata: catalisador novo, vibração de volta ao baseline
    ctx_s2 = ctx.model_copy(
        update={"Catalyst_Age_Days": 0.0, "Vibration_Level_mm_s": 2.0, "Sensor_Health_Index": 0.98}
    )
    # S3 — postergada N dias: degrada até lá, depois reseta
    ctx_s3 = ctx.model_copy(
        update={
            "Catalyst_Age_Days": ctx.Catalyst_Age_Days + N,
            "Vibration_Level_mm_s": min(ctx.Vibration_Level_mm_s * (1 + 0.0015 * N), 20),
        }
    )

    especificacoes = [
        ("S1 — Sem manutenção", ctx_s1, 0.0, 0.0),
        ("S2 — Manutenção imediata", ctx_s2, cfg.CUSTO_MANUTENCAO_PROGRAMADA, cfg.DOWNTIME_MANUTENCAO_H),
        ("S3 — Manutenção postergada", ctx_s3, cfg.CUSTO_MANUTENCAO_PROGRAMADA, cfg.DOWNTIME_MANUTENCAO_H),
    ]

    cenarios = []
    for nome, c, custo_manut, parada in especificacoes:
        r = otimizar(c, body.producao_minima)
        horas_op = cfg.HORIZONTE_DECISAO_H - parada
        p = prob_falha(c, horizonte_h=horas_op)
        margem_horizonte = r["margem"] * horas_op / 4.0  # amostragem do dataset = 4 h
        custo_risco = p * cfg.CUSTO_FALHA_NAO_PROGRAMADA
        cenarios.append(
            {
                "cenario": nome,
                "setpoints": r["setpoints"],
                "producao_ton": r["producao_ton"],
                "energy_intensity": r["energy_intensity"],
                "custo_energia": r["custo_energia"],
                "horas_parada": parada,
                "custo_manutencao": custo_manut,
                "prob_falha": p,
                "custo_esperado_falha": custo_risco,
                "margem_horizonte": margem_horizonte,
                "resultado_liquido": margem_horizonte - custo_manut - custo_risco,
                "condicao_equipamento": {
                    "vibracao": c.Vibration_Level_mm_s,
                    "idade_catalisador": c.Catalyst_Age_Days,
                    "sensor_health": c.Sensor_Health_Index,
                },
            }
        )

    melhor = max(cenarios, key=lambda s: s["resultado_liquido"])
    fazer = not melhor["cenario"].startswith("S1")
    auto = gate_automacao(ctx, melhor["setpoints"])

    return {
        "cenarios": cenarios,
        "recomendacao": {
            "cenario_escolhido": melhor["cenario"],
            "realizar_manutencao": fazer,
            "dias_postergacao_avaliados": N,
            "ganho_vs_sem_manutencao": melhor["resultado_liquido"]
            - next(s["resultado_liquido"] for s in cenarios if s["cenario"].startswith("S1")),
        },
        "automacao": auto,
        "premissas": {
            "custo_manutencao_programada": cfg.CUSTO_MANUTENCAO_PROGRAMADA,
            "custo_falha_nao_programada": cfg.CUSTO_FALHA_NAO_PROGRAMADA,
            "horizonte_h": cfg.HORIZONTE_DECISAO_H,
        },
    }


@app.get("/api/decision")
def decision(
    catalyst_age_days: float = 180,
    sensor_health_index: float = 0.85,
    vibration_level_mm_s: float = 4.5,
    ambient_temp_c: float = 20.0,
) -> dict[str, Any]:
    """Tabela final de decisão (entrega 7 do enunciado), em uma chamada."""
    ART.carregar()
    ctx = Contexto(
        Catalyst_Age_Days=catalyst_age_days,
        Sensor_Health_Index=sensor_health_index,
        Vibration_Level_mm_s=vibration_level_mm_s,
        Ambient_Temp_C=ambient_temp_c,
    )
    manut = maintenance(MaintenanceIn(contexto=ctx))
    esc = next(c for c in manut["cenarios"] if c["cenario"] == manut["recomendacao"]["cenario_escolhido"])
    return {
        "Feedstock Flow": esc["setpoints"]["Feedstock_Flow_m3h"],
        "Reactor Temperature": esc["setpoints"]["Reactor_Temp_C"],
        "Reactor Pressure": esc["setpoints"]["Reactor_Pressure_Bar"],
        "Valve Opening": esc["setpoints"]["Valve_Opening_Percent"],
        "Expected Yield": esc["producao_ton"],
        "Energy Intensity": esc["energy_intensity"],
        "Energy Cost": esc["custo_energia"],
        "Maintenance": "Sim" if manut["recomendacao"]["realizar_manutencao"] else "Não",
        "Maintenance Cost": esc["custo_manutencao"],
        "Total Cost": esc["custo_energia"] + esc["custo_manutencao"] + esc["custo_esperado_falha"],
        "Nivel de automacao recomendado": manut["automacao"]["nivel"],
    }


# ------------------------------------------------------------------ páginas
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/operacao")
def pagina_operacao() -> FileResponse:
    return FileResponse(STATIC / "operacao.html")


@app.get("/manutencao")
def pagina_manutencao() -> FileResponse:
    return FileResponse(STATIC / "manutencao.html")
