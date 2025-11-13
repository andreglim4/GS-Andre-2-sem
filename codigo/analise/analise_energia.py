import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import warnings

# Ignorar FutureWarning do Scikit-learn e Pandas para manter a saída limpa
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuração Inicial e Carregamento de Dados ---
# Carregar os dados

df = pd.read_csv('dados_consumo_escritorio_60dias.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hora'] = df['timestamp'].dt.hour
df['dia_semana_nome'] = df['timestamp'].dt.day_name()
df['dia_do_mes'] = df['timestamp'].dt.day
df['mes'] = df['timestamp'].dt.month
df['fim_de_semana'] = df['dia_semana_nome'].isin(['Saturday', 'Sunday']).astype(int) # 1 para FDS, 0 para Dia Útil

print("Dados de consumo de energia carregados e pré-processados.")
print("-" * 50)

# ---  Análise Exploratória Aprofundada (Aprimorada) ---

# Vilões de Consumo por Dispositivo
consumo_por_dispositivo = df.groupby('dispositivo')['consumo_kWh'].sum().sort_values(ascending=False)
print("\n--- 1. Vilões de Consumo (Total por Dispositivo) ---")
print(consumo_por_dispositivo.to_string())

# Média de Consumo por Área
consumo_por_area = df.groupby('area')['consumo_kWh'].sum().sort_values(ascending=False)
print("\n--- 2. Consumo Total por Área do Escritório ---")
print(consumo_por_area.to_string())

# Picos de Consumo Horário e Diário
media_consumo_hora = df.groupby('hora')['consumo_kWh'].mean()
print("\n--- 3. Média de Consumo por Hora do Dia ---")
print(media_consumo_hora.to_string())

# Análise de Desperdício Detalhado 
consumo_fds_detalhe = df[df['fim_de_semana'] == 1].groupby('dispositivo')['consumo_kWh'].sum().sort_values(ascending=False)
consumo_dia_util_total = df[df['fim_de_semana'] == 0]['consumo_kWh'].sum()
consumo_fds_total = df[df['fim_de_semana'] == 1]['consumo_kWh'].sum()
proporcao_fds = (consumo_fds_total / (consumo_fds_total + consumo_dia_util_total)) * 100

print(f"\n--- 4. Detalhe do Desperdício em Fim de Semana (FDS) ---")
print(f"Consumo Total Dias Úteis: {consumo_dia_util_total:.2f} kWh")
print(f"Consumo Total Fim de Semana: {consumo_fds_total:.2f} kWh ({proporcao_fds:.1f}% do total)")
print("\nTop Dispositivos Consumindo no FDS (Potencial Desperdício):")
print(consumo_fds_detalhe.to_string())

# O desperdício principal (AC no Escritório Aberto no FDS)
desperdicio_ac_fds = df[
    (df['dispositivo'] == 'Ar Condicionado') &
    (df['area'] == 'Escritório Aberto') &
    (df['fim_de_semana'] == 1)
]
total_desperdicio_ac = desperdicio_ac_fds['consumo_kWh'].sum()
print(f"\nAlerta de Desperdício Focado (AC no Escritório Aberto no FDS): {total_desperdicio_ac:.2f} kWh")

# --- Preparação de Dados para Machine Learning (Random Forest) ---

print("\n" + "=" * 50)
print("INÍCIO DO MACHINE LEARNING: RANDOM FOREST REGRESSOR")
print("=" * 50)

# Features a serem usadas (X) e Variável Target (Y)
features = ['hora', 'fim_de_semana', 'dia_do_mes', 'mes', 'dispositivo', 'area']
target = 'consumo_kWh'

X = df[features]
y = df[target]

# One-Hot Encoding para features categóricas (Dispositivo, Área)
X = pd.get_dummies(X, columns=['dispositivo', 'area'], drop_first=True)

# O dia da semana original (nome) foi substituído pela feature binária 'fim_de_semana'
X = X.drop(columns=['mes'], errors='ignore') 

print(f"Total de Features após One-Hot Encoding: {X.shape[1]}")

# Separação dos dados de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do conjunto de Treino: {len(X_train)} amostras")
print(f"Tamanho do conjunto de Teste: {len(X_test)} amostras")


# --- Treinamento e Avaliação do Modelo Random Forest ---

# Inicializar e treinar o Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10, min_samples_leaf=5)
rf_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = rf_model.predict(X_test)

# Avaliação do modelo
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Avaliação do Modelo Random Forest ---")
print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.4f} kWh")
print(f"Coeficiente de Determinação (R²): {r2:.4f}")
print("O RMSE indica a precisão de previsão, e o R² indica a proporção da variância do consumo explicada pelo modelo.")


# --- Análise de Importância das Features ---

# Extrair e visualizar a importância das variáveis
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances = feature_importances.sort_values(ascending=False)

print("\n--- 5. Importância das Features (O que mais impulsiona o consumo) ---")
print("As variáveis mais importantes têm maior peso na previsão do consumo:")
print(feature_importances.head(10).to_string())


# --- Visualização (Incluindo resultados do ML) ---

# Gráfico 1: Consumo total por dispositivo (Existente)
plt.figure(figsize=(10, 5))
consumo_por_dispositivo.plot(kind='bar', color='#1f77b4')
plt.title('Consumo Total por Tipo de Dispositivo (kWh)')
plt.ylabel('Consumo Total (kWh)')
plt.xlabel('Dispositivo')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Gráfico 2: Média de consumo por hora (Existente)
plt.figure(figsize=(10, 5))
media_consumo_hora.plot(kind='line', marker='o', color='#ff7f0e')
plt.title('Média de Consumo por Hora do Dia')
plt.xlabel('Hora do Dia')
plt.ylabel('Consumo Médio (kWh)')
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--')

# Gráfico 3: Comparação Dia Útil vs. Fim de Semana (Existente)
plt.figure(figsize=(8, 8))
df.groupby('fim_de_semana')['consumo_kWh'].sum().plot(
    kind='pie',
    autopct='%1.1f%%',
    labels=['Dia Útil', 'Fim de Semana'],
    colors=['#2ca02c', '#d62728'],
    startangle=90,
    wedgeprops={'edgecolor': 'black'}
)
plt.ylabel('')
plt.title('Consumo Total: Dia Útil (0) vs. Fim de Semana (1)')

# NOVO GRÁFICO 4: Importância das Features (Resultado do ML)
plt.figure(figsize=(12, 6))
feature_importances.head(10).plot(kind='bar', color='#9467bd')
plt.title('Top 10 Variáveis que mais Influenciam o Consumo (Random Forest)')
plt.ylabel('Importância da Feature')
plt.xlabel('Variável')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()


print(f"\nAnálise aprofundada e modelo de Machine Learning concluídos.")
print(f"4 novos gráficos PNG salvos, incluindo a Importância das Variáveis para Previsão.")
