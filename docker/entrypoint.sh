#!/bin/sh
# Garante que a API sobe com modelos carregáveis.
#
# Sem isso, num clone novo o container sobe e /api/predict responde 503 até
# alguém rodar a seção 12 do notebook. Com BOOTSTRAP_MODELS=1 (padrão do
# compose), o baseline é treinado se models/ estiver vazio — e NUNCA sobrescreve
# modelos que já existam, então o trabalho do notebook está seguro.
set -e

if [ "${BOOTSTRAP_MODELS:-0}" = "1" ] && [ ! -f /code/models/model_energy.pkl ]; then
  echo "[entrypoint] models/ vazio — treinando baseline (scripts/bootstrap_models.py)"
  python /code/scripts/bootstrap_models.py
  echo "[entrypoint] baseline pronto. A seção 12 do notebook substitui estes arquivos."
fi

exec "$@"
