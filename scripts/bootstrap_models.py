"""
Treina modelos BASELINE e salva em models/, só para a aplicação web subir
funcionando antes de o notebook estar pronto.

NÃO é a entrega. São dois Ridge simples, sem validação temporal decente, sem
comparação de modelos e sem tuning. O notebook (seções 4–5) é que define os
modelos de verdade e sobrescreve estes arquivos na seção 12.

Uso:  .venv\\Scripts\\python.exe scripts\\bootstrap_models.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.config as cfg

FEATURES = cfg.CONTROLAVEIS + cfg.ESTADO + [
    "Ambient_Temp_C", "Delta_Temp", "Severidade", "Carga_por_Abertura", "Idade_Norm",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["Delta_Temp"] = X["Reactor_Temp_C"] - X["Ambient_Temp_C"]
    X["Severidade"] = X["Reactor_Temp_C"] * X["Reactor_Pressure_Bar"]
    X["Carga_por_Abertura"] = X["Feedstock_Flow_m3h"] / X["Valve_Opening_Percent"].replace(0, np.nan)
    X["Idade_Norm"] = X["Catalyst_Age_Days"] / cfg.LIMITE_IDADE_CATALISADOR
    return X


def main() -> None:
    df = pd.read_csv(cfg.DATA_RAW, parse_dates=["Timestamp"]).sort_values("Timestamp")
    df = build_features(df).dropna(subset=FEATURES + cfg.TARGETS)

    corte = int(len(df) * 0.8)                       # split temporal, não aleatório
    Xtr, Xte = df[FEATURES].iloc[:corte], df[FEATURES].iloc[corte:]

    metricas, modelos = [], {}
    for target, arquivo in [("Energy_Intensity", "model_energy.pkl"),
                            ("Product_Yield_Tons", "model_yield.pkl")]:
        ytr, yte = df[target].iloc[:corte], df[target].iloc[corte:]
        mdl = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]).fit(Xtr, ytr)
        p = mdl.predict(Xte)
        metricas.append({
            "Target": target, "Modelo": "Ridge (baseline)",
            "R2": r2_score(yte, p),
            "RMSE": mean_squared_error(yte, p) ** 0.5,
            "MAE": mean_absolute_error(yte, p),
            "MAPE_%": float(np.mean(np.abs((yte - p) / yte)) * 100),
        })
        cfg.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(mdl, cfg.MODELS_DIR / arquivo)
        modelos[target] = mdl

    bounds = {c: [float(np.percentile(df[c], 5)), float(np.percentile(df[c], 95))]
              for c in cfg.CONTROLAVEIS}
    ctx = {c: float(df[c].tail(24).median()) for c in cfg.ESTADO + ["Ambient_Temp_C"]}

    meta = {
        "features": FEATURES,
        "bounds": bounds,
        "contexto_default": ctx,
        "metricas": metricas,
        "premissas": {
            "preco_produto_ton": cfg.PRECO_PRODUTO_TON,
            "custo_energia_unidade": cfg.CUSTO_ENERGIA_POR_UNIDADE_INTENSIDADE,
            "custo_manutencao": cfg.CUSTO_MANUTENCAO_PROGRAMADA,
            "custo_falha": cfg.CUSTO_FALHA_NAO_PROGRAMADA,
            "producao_minima": cfg.PRODUCAO_MINIMA_TON,
        },
        "aviso": "Modelos BASELINE gerados por scripts/bootstrap_models.py — substituir pela seção 12 do notebook.",
    }
    (cfg.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(pd.DataFrame(metricas).to_string(index=False))
    print("\nArtefatos salvos em", cfg.MODELS_DIR)


if __name__ == "__main__":
    main()
