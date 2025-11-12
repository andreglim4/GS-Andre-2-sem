import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados
df = pd.read_csv('dados_consumo_escritorio_60dias.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hora'] = df['timestamp'].dt.hour
df['dia_semana'] = df['timestamp'].dt.day_name()
df['dia_do_mes'] = df['timestamp'].dt.day
df['fim_de_semana'] = df['dia_semana'].isin(['Saturday', 'Sunday'])

print("Dados carregados. Iniciando análise...")

# 1. Vilões de Consumo: Qual dispositivo gasta mais?
consumo_por_dispositivo = df.groupby('dispositivo')['consumo_kWh'].sum().sort_values(ascending=False)
print("\n--- Vilões de Consumo (Total) ---")
print(consumo_por_dispositivo)

# 2. Picos de Consumo: Em que horas o consumo é maior?
consumo_por_hora = df.groupby('hora')['consumo_kWh'].mean()
print("\n--- Média de Consumo por Hora ---")
print(consumo_por_hora)

# 3. IDENTIFICANDO DESPERDÍCIO: Consumo em Fim de Semana
consumo_fds = df[df['fim_de_semana'] == True]
consumo_dia_util = df[df['fim_de_semana'] == False]

print(f"\nConsumo Total Fim de Semana: {consumo_fds['consumo_kWh'].sum():.2f} kWh")
print(f"Consumo Total Dias Úteis: {consumo_dia_util['consumo_kWh'].sum():.2f} kWh")

# Foco no desperdício principal (Ar Condicionado no FDS)
desperdicio_ac_fds = consumo_fds[
    (consumo_fds['dispositivo'] == 'Ar Condicionado') & 
    (consumo_fds['area'] == 'Escritório Aberto')
]
total_desperdicio_ac = desperdicio_ac_fds['consumo_kWh'].sum()
print(f"Total de Desperdício (AC no Escritório Aberto no FDS): {total_desperdicio_ac:.2f} kWh")

# --- Visualização (Essencial para a apresentação) ---

# Gráfico 1: Consumo total por dispositivo
plt.figure(figsize=(10, 5))
consumo_por_dispositivo.plot(kind='bar', color='red')
plt.title('Consumo Total por Tipo de Dispositivo')
plt.ylabel('Consumo Total (kWh)')
plt.savefig('grafico_consumo_dispositivo.png')

# Gráfico 2: Média de consumo por hora (identifica picos)
plt.figure(figsize=(10, 5))
consumo_por_hora.plot(kind='line', marker='o')
plt.title('Média de Consumo por Hora do Dia')
plt.xlabel('Hora')
plt.ylabel('Consumo Médio (kWh)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.savefig('grafico_consumo_hora.png')

# Gráfico 3: Comparação Dia Útil vs. Fim de Semana
df.groupby('fim_de_semana')['consumo_kWh'].sum().plot(
    kind='pie', 
    autopct='%1.1f%%', 
    labels=['Dia Útil', 'Fim de Semana'],
    title='Consumo: Dia Útil vs. Fim de Semana'
)
plt.savefig('grafico_consumo_fds.png')

print(f"\nAnálise concluída. Gráficos salvos em PNG.")

