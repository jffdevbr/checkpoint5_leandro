# Trabalho de Pesquisa — Pipeline de ML, Otimização e Decisão de Manutenção

## Objetivo
Com base na ultima aula de otimização, vamos fazer um trabalho que vai além da otimização, é um pipeline completo e que demanda uma decisão e pesquisa de vocês. Essa entrega deve ser feita em sala de aula, durante esta aula e a próxima com entrega para próxima aula no dia 24/09. Pode ser realizada individualmente ou em grupo.

Utilizando o dataset petrochemical_advanced_data.csv, construir um pipeline que transforme dados históricos em uma decisão operacional, combinando:

Dados → Machine Learning → Otimização → Cenários → Decisão

O trabalho deve responder:

Qual configuração operacional reduz o custo/intensidade energética mantendo a produção e as restrições operacionais?

E também:

Devemos realizar manutenção? Essa decisão poderia ser automatizada?

## Entregas obrigatórias

### 1. Entendimento do negócio
Identificar e documentar:

- objetivo da operação;
- variáveis controláveis;
- variáveis de estado;
- variáveis externas;
- principais restrições;
- regras de negócio necessárias.

Apresentar uma tabela com as principais variáveis e sua classificação.

### 2. Machine Learning
Criar modelos para prever:

- Energy_Intensity
- Product_Yield_Tons

Utilizar pelo menos 2 modelos para cada target.

Apresentar uma tabela comparando os modelos:

Energy	Modelo 1			
Energy	Modelo 2			
Yield	Modelo 1			
Yield	Modelo 2			

Explicar qual modelo foi utilizado no pipeline e por quê.
Importante: verificar possível target leakage, especialmente nas variáveis de energia.

### 3. Otimização
Transformar os modelos preditivos em um problema de otimização.

Definir:

- variáveis de decisão;
- função objetivo;
- restrições;
- limites das variáveis.

Implementar utilizando scipy.optimize.linprog ou outro solver justificado.

Apresentar:

- formulação matemática;
- código;
- solução encontrada;
- interpretação da solução.

### 4. Decisão de manutenção
Criar pelo menos 3 cenários:

1. Sem manutenção
2. Manutenção imediata
3. Manutenção postergada

Comparar:

- produção;
- energia;
- custo;
- condição do equipamento;
- risco;
- impacto esperado.

### 5. Automação da decisão
Analisar criticamente:

A decisão de manutenção deve ser automática ou passar por aprovação humana?

Apresentar:

1. Benefícios da automação

- velocidade;
- escala;
- redução de intervenção;
- redução de downtime/custos;
- operação contínua.

2. Riscos

- falso positivo;
- falso negativo;
- erro do modelo;
- dados incorretos;
- mudança das condições operacionais;
- impacto de uma decisão errada.

Explicar qual nível de automação vocês recomendariam e justificar.

### 6. Pipeline final
Entregar uma representação do pipeline completo:

Dados   ↓ EDA   ↓ Feature Engineering   ↓ Modelos ML   ↓ Previsões   ↓ Cenários   ↓ Otimização   ↓ Análise de custo/risco   ↓ Decisão de manutenção   ↓ Recomendação   ↓ Humano ou Automação

### 7. Resultado final
Apresentar uma tabela final de decisão contendo:

Feedstock Flow	
Reactor Temperature	
Reactor Pressure	
Valve Opening	
Expected Yield	
Energy Intensity	
Energy Cost	
Maintenance	Sim/Não
Maintenance Cost	
Total Cost	
Nível de automação recomendado	

### 8. Conclusão
Responder objetivamente:

- Qual configuração operacional foi recomendada?
- A manutenção deve ser realizada? Em quais condições?
- Qual foi o impacto econômico esperado?
- Quais são os principais riscos da decisão?
- A decisão deve ser automatizada ou aprovada por um humano?
- Quais dados adicionais seriam necessários para colocar essa solução em produção?

## Entregáveis
1. Notebook com todo o pipeline executável.
2. Relatório explicando metodologia, resultados e decisões.
3. Apresentação de aproximadamente 10–15 minutos.

O foco não é apenas obter o melhor modelo ou a solução matemática. O objetivo é demonstrar como ML + Otimização podem ser transformados em uma decisão operacional justificável.