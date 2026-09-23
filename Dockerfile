# Imagem única para os dois serviços do compose (API e Jupyter) — as duas
# precisam das mesmas dependências e do mesmo src/, então manter uma imagem
# evita divergência entre o que o notebook treina e o que a API serve.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/code

# libgomp1 -> requisito de runtime do lightgbm e do xgboost
# curl      -> usado pelo HEALTHCHECK
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Dependências primeiro, em camada própria: mexer no código não reinstala nada.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY notebooks/ ./notebooks/
COPY data/raw/ ./data/raw/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh \
 && mkdir -p models data/processed reports/figures

# Roda como root de propósito: models/ e reports/ são bind mounts do host, e um
# usuário não-root do container não teria permissão de escrever neles em hosts
# Linux. É aceitável aqui porque a aplicação só é exposta em localhost.

EXPOSE 8000 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
