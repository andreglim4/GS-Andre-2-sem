# GS: SOLUÇÕES EM ENERGIAS RENOVÁVEIS E SUSTENTÁVEIS - 2°semestre - 1CCPG  
André Ayello de Nobrega: rm561754 --
André Gouveia de Lima: rm564219 -- 
Mirella Mascarenhas: rm562092 

# SmartOffice - Gestão Inteligente de Energia

Este projeto propõe uma solução de eficiência energética para ambientes de trabalho modernos, focada na otimização de consumo e automação.

## Objetivo

Analisar dados simulados de consumo de energia de um escritório comercial para identificar desperdícios e propor uma solução baseada em IoT (Internet das Coisas) para automação e controle, promovendo sustentabilidade e redução de custos.

## A Solução (Opção A e B)

Nossa solução combina Análise de Dados (Opção A) com a simulação de um Dispositivo IoT (Opção B).

1.  **Análise de Dados (A):** Utilizamos um dataset simulado (localizado em `/dados/`) para identificar padrões de consumo. A análise (ver `/analise/`) revelou um grande desperdício de energia com Ar Condicionado nos fins de semana, quando o escritório está vazio.
2.  **Solução IoT (B):** Propomos um sistema de automação (simulado em `/codigo/simulador_iot_automacao.py`) que integra sensores (presença, horário) para desligar dispositivos automaticamente, corrigindo o desperdício encontrado na análise.

### Resultados da Análise

A análise dos dados identificou que o **Ar Condicionado** é o principal vilão do consumo.

**Gráfico 1: Consumo Total por Dispositivo**
![Consumo por Dispositivo](./analise/grafico_consumo_dispositivo.png)

O maior desperdício ocorre nos fins de semana, como visto no gráfico de comparação, onde quase 15% do consumo total ocorre em períodos sem expediente.

**Gráfico 2: Consumo Dia Útil vs. Fim de Semana**
![Consumo Fim de Semana](./analise/grafico_consumo_fds.png)

A análise horária confirma que os picos de consumo ocorrem no horário comercial (08h-18h), mas um consumo residual alto (desperdício) permanece fora desse horário.

**Gráfico 3: Média de Consumo por Hora**
![Consumo por Hora](./analise/grafico_consumo_hora.png)

## Conexão com o Futuro do Trabalho

Esta solução contribui para o futuro do trabalho ao criar ambientes:
* **Eficientes e Econômicos:** Redução direta na conta de energia através da automação inteligente, eliminando o desperdício.
* **Sustentáveis (ESG):** Diminuição da pegada de carbono da empresa, alinhando-se a metas ESG (Environmental, Social, and Governance).
* **Inteligentes e Responsivos:** O ambiente de trabalho se adapta ao uso real, desligando sistemas em salas vazias e otimizando o conforto apenas quando necessário.
* **Baseado em Dados:** A gestão do escritório passa a ser orientada por dados (Data-Driven), permitindo melhor alocação de recursos.

## Estrutura do Repositório

/Global-Solution-SmartOffice
|
README.md             (Este arquivo)
    (dados)
    dados_consumo_escritorio.csv  (Dados simulados)

    (codigo)
    Gerador_dados.py
    simulador_iot_automacao.py

    (analise)
    analise_energia.py
    grafico_consumo_dispositivo.png (Resultado da análise)
    grafico_consumo_hora.png        (Resultado da análise)
    grafico_consumo_fds.png         (Resultado da análise)
