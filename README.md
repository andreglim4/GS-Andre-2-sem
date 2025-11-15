# GS: SOLUÇÕES EM ENERGIAS RENOVÁVEIS E SUSTENTÁVEIS - 2°semestre - 1CCPG  
André Ayello de Nobrega: rm561754 --
André Gouveia de Lima: rm564219 -- 
Mirella Mascarenhas: rm562092 

# SmartOffice - Gestão Inteligente de Energia

Este projeto propõe uma solução de eficiência energética para ambientes de trabalho modernos, focada na otimização de consumo e automação.

## Objetivo

Analisar dados simulados de consumo de energia de um escritório comercial para identificar desperdícios e propor uma solução baseada em IoT (Internet das Coisas) para automação e controle, promovendo sustentabilidade e redução de custos.

## A Solução (Opção A)

Nossa solução combina Análise de Dados (Opção A) 

1.  **Análise de Dados (A):** Utilizamos um dataset simulado (localizado em `/dados/`) para identificar padrões de consumo. A análise (ver `/analise/`) revelou um grande desperdício de energia com Ar Condicionado nos fins de semana, quando o escritório está vazio.

## Resultados da Análise

📈 **Análise Exploratória:** Identificando o Desperdício

A primeira fase da análise focou em transformar dados brutos em insights de gestão:

Vilões de Consumo: O Ar Condicionado e o Servidor foram identificados como os maiores consumidores de kWh.

Padrão de Uso: O pico de consumo ocorre durante o horário comercial (meio da tarde), mas o consumo de base (dispositivos 24/7) é relevante.

Alerta de Desperdício: A análise detalhada em Finais de Semana (FDS) revelou um consumo atípico e alto do Ar Condicionado na área do Escritório Aberto, um claro indicativo de falha na automação ou no desligamento manual. Este é o principal alvo de otimização.

🤖 **Solução Preditiva:** Machine Learning com Random Forest

Para construir uma ferramenta de gestão ativa, desenvolvemos uma arquitetura de ML duplo, ambos baseados na robustez do algoritmo Random Forest:

1. Modelo de Regressão (Random Forest Regressor)

O Regressor tem como propósito principal prever o valor contínuo e exato do consumo em kWh (consumo_kWh) em qualquer hora.

A performance do modelo é extremamente alta. O Coeficiente R² se aproxima de 0.998, o que confirma que o modelo explica quase toda a variância do consumo, sendo altamente preditivo. O RMSE (Raiz do Erro Quadrático Médio) é muito baixo, em torno de 0.117 kWh, validando a precisão da previsão.

As variáveis que mais impulsionam o consumo (Top Features) são, em ordem decrescente de importância: o dispositivo_Ar Condicionado, seguido pelo dispositivo_Servidor e, em terceiro lugar, a hora do dia.

2. Modelo de Classificação (Random Forest Classifier)

O Classifier foi desenhado para classificar o consumo de energia em um dado momento como ALTO (1) ou NORMAL (0). O limite para "Alto" foi estabelecido usando o 75º percentil (Q3) do consumo histórico.

A performance de classificação é notável, com a Acurácia atingindo quase 0.999. Isso significa que o modelo é excelente em prever quando um momento terá um consumo anormalmente alto ou não. Tanto a Precisão quanto o Recall são altos para ambas as classes, demonstrando que a identificação de picos de consumo é robusta.

As variáveis cruciais para classificar o consumo como ALTO são as mesmas do Regressor: o dispositivo_Ar Condicionado, o dispositivo_Servidor e a hora do dia.

💡 **Ganhos, Sustentabilidade e Futuro do Trabalho**

A integração dos modelos ML com a gestão do escritório gera impactos significativos, transformando a ineficiência em ação sustentável:

Ganho Econômico (Ação Corretiva Focada)

O Modelo de Regressão permite estimar a economia exata em kWh que será alcançada ao eliminar o desperdício do Ar Condicionado no FDS, fornecendo uma base sólida para o ROI (Retorno sobre o Investimento) de medidas corretivas. Esta é a via mais rápida para o ganho imediato.

Ganho Operacional (Detecção de Anomalias em Tempo Real)

O Modelo de Classificação previne o desperdício e falhas operacionais ao alertar instantaneamente sobre picos de consumo ALTO em horários não usuais (e.g., madrugadas ou FDS). Isso evita falhas de automação e o consumo indevido por dispositivos que deveriam estar inativos.

Ganho Sustentável (Automação Preditiva)

Promove práticas de sustentabilidade avançadas, essenciais para o futuro do trabalho. Sistemas de energia (AC, Iluminação) podem ser programados para ligar ou desligar baseados na previsão de necessidade e ocupação (utilizando o output do Classifier), e não apenas em horários fixos. Isso garante que a energia seja consumida apenas quando e onde é realmente necessária.

## Como Executar o Código

O código pode ser executado através do arquivo analise_energia.py:

python analise_energia.py


Pré-requisitos: Python 3.x e as bibliotecas Pandas, Matplotlib, Seaborn e Scikit-learn instaladas.

pip install pandas matplotlib seaborn scikit-learn numpy


O arquivo de dados (dados_consumo_escritorio_60dias.csv) deve estar no mesmo diretório.
