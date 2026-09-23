# Checkpoint 5 — Pipeline de ML, Otimização e Decisão de Manutenção

Dados → Machine Learning → Otimização → Cenários → Decisão, sobre
`petrochemical_advanced_data.csv` (10.000 registros de 4 em 4 h, 3 unidades).

Comece por **[PLANO.md](PLANO.md)** — é o mapa do trabalho.

## Rodar com Docker (recomendado para a apresentação)

Não precisa de Python nem do venv na máquina — só do Docker Desktop aberto.

```powershell
docker compose up --build        # sobe API + Jupyter
docker compose up api            # sobe só a API
docker compose down              # derruba
```

| Serviço | Endereço | O que é |
|---|---|---|
| `api` | <http://127.0.0.1:8000> | páginas de decisão + endpoints do modelo (`/docs` para a API) |
| `jupyter` | <http://127.0.0.1:8888> | JupyterLab com o notebook do pipeline (sem token) |

Os dois serviços compartilham a mesma imagem e montam `models/` do host: o que o notebook
treina na seção 12 é imediatamente o que a API serve, **sem rebuild**. `src/` e `app/` também
são montados, e o uvicorn sobe com `--reload` — editar código reflete na hora.

Se `models/` estiver vazio, o entrypoint treina o baseline automaticamente
(`BOOTSTRAP_MODELS=1` no compose). Ele **nunca** sobrescreve modelos existentes.

As portas escutam só em `127.0.0.1`, e é por isso que o Jupyter roda sem token. Se você
publicar em outra interface, reative o token no `docker-compose.yml`.

## Setup local (sem Docker)

O ambiente já está criado. Para ativar:

```powershell
.venv\Scripts\activate
```

Recriar do zero, se precisar:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name checkpoint5 --display-name "Python (checkpoint5)"
```

```powershell
jupyter lab                       # notebook — selecionar o kernel "Python (checkpoint5)"
uvicorn app.main:app --reload     # aplicação web
```

App em <http://127.0.0.1:8000> · documentação automática da API em `/docs`.

## Estrutura

```
├── PLANO.md                     roteiro, ideias por entregável, divisão de tarefas
├── instrucoes.md                enunciado
├── Dockerfile                   imagem única usada pelos dois serviços
├── docker-compose.yml           serviços api (8000) e jupyter (8888)
├── docker/entrypoint.sh         treina baseline se models/ estiver vazio
├── notebooks/
│   └── 01_pipeline_completo.ipynb   esqueleto das 12 seções (células # TODO = trabalho do grupo)
├── src/
│   └── config.py                classificação de variáveis + premissas econômicas (fonte da verdade)
├── scripts/
│   └── bootstrap_models.py      modelos baseline, só para o app subir antes do notebook pronto
├── models/                      model_energy.pkl · model_yield.pkl · metadata.json
├── app/
│   ├── main.py                  FastAPI: /api/predict, /api/optimize, /api/maintenance, /api/decision
│   └── static/                  index · operacao · manutencao
├── data/raw/                    dataset
└── reports/                     relatório e figuras
```

## Endpoints

| Rota | O que faz |
|---|---|
| `POST /api/predict` | setpoints + contexto → intensidade energética, produção, custo, margem |
| `POST /api/optimize` | menor custo mantendo produção ≥ mínima, dentro da faixa de treino |
| `POST /api/maintenance` | compara S1/S2/S3, calcula risco e devolve a recomendação |
| `GET /api/decision` | tabela final de decisão (entrega 7 do enunciado) |
| `GET /api/metadata` | variáveis, limites, premissas e métricas dos modelos |

## Páginas

- `/` — visão geral, diagrama do pipeline, achados e status dos modelos
- `/operacao` — simulador: mexa nos setpoints e veja a previsão do modelo ao vivo; botão de otimizar
- `/manutencao` — cenários, risco, recomendação e nível de automação permitido

## Atenção

Os modelos em `models/` hoje são **baseline** (`scripts/bootstrap_models.py`), presentes só para o
app funcionar. A seção 12 do notebook sobrescreve esses arquivos com os modelos da entrega.

As premissas de preço e custo em `src/config.py` **não vêm do dataset** — foram assumidas pelo grupo.
Toda conclusão em R$ depende delas e elas precisam aparecer no relatório e na apresentação.