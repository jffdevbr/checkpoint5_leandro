# Plano de trabalho — Checkpoint 5

Entrega: **24/09**. Este arquivo é o mapa: o que fazer, em que ordem, e as ideias
que diferenciam cada entregável.

---

## 0. O que já está pronto

| Item | Onde |
|---|---|
| Ambiente virtual + libs | `.venv/`, `requirements.txt` |
| Kernel Jupyter registrado | `Python (checkpoint5)` |
| Esqueleto do notebook (48 células, 12 seções) | `notebooks/01_pipeline_completo.ipynb` |
| Classificação de variáveis + premissas econômicas | `src/config.py` |
| Modelos baseline (para o app subir hoje) | `scripts/bootstrap_models.py` → `models/` |
| API + 3 páginas web, testadas ponta a ponta | `app/` |
| Container + compose (API na 8000, Jupyter na 8888) | `Dockerfile`, `docker-compose.yml` |

```powershell
docker compose up --build                    # tudo em container
```

ou, local:

```powershell
.venv\Scripts\activate
jupyter lab                                  # notebook
uvicorn app.main:app --reload                # app em http://127.0.0.1:8000
```

> Para a apresentação, o Docker é a aposta mais segura: se o notebook de alguém quebrar
> na hora, `docker compose up` sobe o pipeline inteiro em qualquer máquina da sala.

---

## 1. Os dois achados que devem guiar o trabalho

Ambos verificados nos dados, não assumidos. Estão no notebook (seção 4) e na página inicial do app.

**(a) Target leakage confirmado — R² = 1,000000**

```
Energy_Intensity = (3,6·Electricity_MWh + 0,035·Natural_Gas_m3h) / Product_Yield_Tons
```

As colunas de consumo não são features, são o target reescrito. O enunciado pede
explicitamente essa verificação — é ponto garantido, e a demonstração "modelo com
leakage (R²≈1) vs. modelo honesto" é o gráfico mais convincente da apresentação.

**(b) A alavanca real não é o setpoint — R² = 1,000000**

```
Product_Yield_Tons = 0,18 · Feedstock_Flow_m3h · Sensor_Health_Index
```

`Reactor_Temp_C`, `Reactor_Pressure_Bar` e `Valve_Opening_Percent` têm correlação ≈ 0
com os dois targets, inclusive dentro de cada unidade. A única alavanca operacional é a
vazão; e a condição do equipamento multiplica a produção diretamente.

> **A narrativa do trabalho sai daí:** otimizar setpoints rende pouco porque quase não
> há setpoint que importe. **A decisão que move o resultado é a de manutenção.** Isso
> conecta as duas perguntas do enunciado numa tese só, em vez de duas seções soltas.
>
> Seja explícito sobre a limitação: o dataset é sintético; numa planta real temperatura
> e pressão teriam efeito. Dizer isso é sinal de maturidade analítica, não de fraqueza.

---

## 2. Roteiro por entregável

### 1 — Entendimento do negócio
- A tabela de variáveis é **gerada por código** a partir de `src/config.py` (seção 1.2), então
  a classificação não é decorativa: ela é a mesma que o otimizador e a API usam.
- Ideia forte: enquadrar o objetivo como **maximizar margem sob restrição**, não "minimizar
  energia". Minimizar `Energy_Intensity` isolado tem solução trivial e errada — produzir pouco.
  Mostrar essa armadilha e resolvê-la com a restrição de produção mínima já vale o capítulo.
- Escreva as regras de negócio (R1–R5 na seção 1.3) **já pensando em como cada uma vira código**.
  A R4 (`Sensor_Health_Index` baixo ⇒ decisão não pode ser automática) é a que amarra a seção 1
  na seção 5 do enunciado.

### 2 — Machine Learning
- Dois modelos por target, como pedido: **Ridge** (interpretável, dá gradiente, permite formulação LP)
  e **Gradient Boosting / XGBoost** (captura não linearidade, serve de validação cruzada de sanidade).
- **Split temporal, não aleatório.** São 10.000 registros de 4 em 4 horas. `train_test_split`
  aleatório vaza futuro no passado. Usar corte por data ou `TimeSeriesSplit` — e *dizer* que usou.
- **Critério de escolha do modelo do pipeline:** não é o maior R². O modelo vai ser chamado dentro
  do otimizador, então precisa de erro baixo **e** de ser suave (árvore é constante por partes:
  gradiente zero, o SLSQP trava) **e** rápido. Justificar assim rende mais que uma tabela de métricas.
- Baseline já medido: Ridge dá R² 0,677 em `Energy_Intensity` e 0,992 em `Product_Yield_Tons`.
  Se o boosting não superar isso de forma relevante, **diga** — "o modelo simples bastou" é uma
  conclusão legítima e bem vista.

### 3 — Otimização
- Faça **as duas rotas** e compare; custa pouco e mostra domínio:
  - **LP com `scipy.optimize.linprog`** — o que a aula cobriu, auditável, formulação bonita no relatório.
  - **Não linear com `minimize`/SLSQP** — necessário porque o custo envolve o produto
    `Energy_Intensity × Yield`, que é não linear no espaço de decisão. Essa é a justificativa
    que o enunciado pede ao dizer "ou outro solver justificado".
- **Checklist de sanidade (seção 6.6)** — é aqui que se ganha nota. Na rodada de teste, o otimizador
  encravou 3 das 4 variáveis nos limites. Solução no canto = o modelo está sendo extrapolado.
  Reconhecer e discutir isso vale mais do que apresentar o número como se fosse verdade.
- **Degenerescência:** como temperatura/pressão/válvula não afetam nada, há infinitas soluções ótimas.
  Não apresente valores arbitrários como recomendação. Proponha um critério de desempate defensável
  (ex.: entre as soluções ótimas, a mais próxima da operação atual = menor perturbação).

### 4 — Cenários de manutenção
- O ponto central: **manutenção é uma troca**, não um custo. Custa parada + serviço e devolve
  eficiência (`Sensor_Health_Index` restaurado ⇒ produção maior) e reduz risco de falha.
- Cada cenário **re-otimiza a operação** com o estado de equipamento que ele implica. Sem isso,
  os cenários não são comparáveis — já está implementado assim em `app/main.py::maintenance`.
- Coloque risco em reais: `E[Custo] = operação + P(falha)·custo_falha + manutenção`. Aí S1/S2/S3
  viram três números comparáveis e a decisão deixa de ser opinião.
- **O gráfico do slide principal:** varrer a postergação de 0 a 60 dias e plotar o custo esperado.
  O mínimo da curva é o **ponto de indiferença** — a resposta quantitativa para "em quais condições
  fazer a manutenção".

### 5 — Automação da decisão
- Traduza a matriz de confusão **em dinheiro**. Com as premissas de `config.py`, um falso negativo
  (quebra) custa ~6,7× um falso positivo (parada desnecessária). Logo **o limiar ótimo não é 0,5** —
  calcule qual é. Resposta quantitativa, não retórica.
- **Recomendação sugerida — automação assimétrica por reversibilidade:**
  **L2** para ajuste de setpoints (reversível, erro barato, ganho contínuo) e
  **L1** para a ordem de manutenção (irreversível, caro, envolve segurança).
- Mais os *gates* que rebaixam para L0 automaticamente — já implementados em
  `gate_automacao()` e visíveis na página de manutenção:
  sensor degradado · ponto fora da faixa de treino · discordância entre modelos · drift.
- **A tese:** o nível de automação não é propriedade do sistema, é **função da confiança do modelo
  naquele ponto específico de operação**. Um sistema que sabe quando não sabe pode ser mais
  automatizado do que um que sempre responde.

### 6 e 7 — Pipeline e tabela final
- Diagrama pronto no notebook (seção 9) e na home do app. O detalhe que vale nota é a **seta de volta**:
  a manutenção muda o estado, que muda o modelo, que muda a operação ótima.
- A tabela final do enunciado está implementada como endpoint: `GET /api/decision`. Chame na
  apresentação ao vivo — a tabela sai pronta e mostra que o pipeline roda de verdade.
- Sempre ao lado da linha de base (operação média histórica), para o ganho aparecer em R$.

### 8 — Conclusão
Responda as 6 perguntas de forma direta, uma por parágrafo. A de "dados adicionais" é a mais fácil
de brilhar, e as respostas fortes aqui são:
- **histórico real de falhas e ordens de manutenção** — sem isso o modelo de risco é suposição;
- **preços reais** de energia (horários/contrato) e do produto — hoje são premissa nossa;
- **limites de segurança oficiais** (P&ID, HAZOP) em vez de percentis históricos;
- **dados de variação deliberada de setpoints** — dados observacionais limitam inferência causal,
  e otimização é intrinsecamente causal. Este é o argumento mais sofisticado disponível.

---

## 3. Apresentação (10–15 min)

Sugestão de roteiro — **10 slides**, um argumento cada:

| # | Slide | Mensagem |
|---|---|---|
| 1 | As duas perguntas | por que este pipeline existe |
| 2 | Variáveis e restrições | a tabela de classificação |
| 3 | **Leakage** | R²≈1 vs. modelo honesto — o gráfico do contraste |
| 4 | **A alavanca real** | setpoints não movem; condição do equipamento move |
| 5 | Modelos | tabela comparativa + por que escolhemos este |
| 6 | Formulação | objetivo, restrições, limites |
| 7 | Solução | configuração ótima + ressalva da degenerescência |
| 8 | **Cenários** | curva do custo esperado e o ponto de indiferença |
| 9 | Automação | matriz de erro em R$, limiar ≠ 0,5, níveis + gates |
| 10 | **Demo ao vivo** | abrir `/manutencao`, mexer na vibração, ver a decisão virar |

O slide 10 é o diferencial: quase ninguém leva pipeline rodando. Mexa o
`Sensor_Health_Index` abaixo de 0,70 na frente da turma e mostre a recomendação
cair de L2 para L0 com o motivo escrito na tela.

---

## 4. Divisão sugerida (se for em grupo)

| Frente | Entregas | Arquivos |
|---|---|---|
| Dados & ML | 1, 2 | notebook seções 1–5 |
| Otimização | 3, 6 | notebook seções 6, 9 |
| Decisão & risco | 4, 5, 7 | notebook seções 7, 8, 10 |
| Relatório & apresentação | 8 + slides | `reports/`, app |

Ponto de sincronização: `src/config.py`. Todo mundo lê de lá — quem mudar premissa, avisa.

---

## 5. Ordem de execução recomendada

1. Rodar o notebook inteiro como está, para ver o esqueleto funcionando.
2. Fechar a seção 4 (leakage) — é a que sustenta todo o resto.
3. Seção 5 (modelos) e decidir qual entra no pipeline.
4. Seção 12 (export) → os modelos de verdade substituem os baseline e o app passa a refletir o trabalho.
5. Seções 6 e 7 (otimização e cenários), que é onde está o conteúdo analítico.
6. Seções 8 e 11 (automação e conclusão) — escrever depois de ver os números.
7. Relatório e slides por último, puxando das saídas do notebook.
