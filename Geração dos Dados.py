import pandas as pd
import numpy as np
import random

# Configuração da simulação
dias = 30
horas_por_dia = 24
n_registros = dias * horas_por_dia
areas = ['Escritório Aberto', 'Sala Reunião A', 'Sala Reunião B', 'Copa', 'Recepção']
dispositivos_base = {
    'Iluminação': 1.5,  # kWh base
    'Ar Condicionado': 4.0, # kWh base
    'PCs e Monitores': 2.5  # kWh base
}

data = []
# Criar um range de datas
timestamps = pd.date_range(start='2025-10-01', periods=n_registros, freq='h')

for ts in timestamps:
    hora = ts.hour
    dia_semana = ts.dayofweek  # 0-Seg, 1-Ter, ..., 6-Dom

    for area in areas:
        for dispositivo, consumo_base in dispositivos_base.items():
            
            # Lógica de simulação de desperdício e uso
            consumo = 0
            
            # Horário comercial (8h-18h) e dias de semana (0-4)
            if 8 <= hora <= 18 and dia_semana < 5:
                consumo = consumo_base + random.uniform(-0.5, 1.5)
            # Consumo reduzido fora do horário
            elif 18 < hora < 23 and dia_semana < 5:
                 consumo = consumo_base * 0.3 + random.uniform(0, 0.5)
            # DESPERDÍCIO: Consumo residual/alto em fins de semana ou madrugada
            elif dia_semana >= 5 or hora < 6:
                if area == 'Escritório Aberto' and dispositivo == 'Ar Condicionado':
                    # AC ligado sem ninguém no fim de semana
                    consumo = consumo_base * 0.8 + random.uniform(0, 0.5) 
                else:
                    consumo = consumo_base * 0.1 + random.uniform(0, 0.1) # Standby
            
            # Garantir que não seja negativo
            consumo = max(0, consumo)

            data.append({
                'timestamp': ts,
                'area': area,
                'dispositivo': dispositivo,
                'consumo_kWh': round(consumo, 2)
            })

# Criar DataFrame e salvar em CSV
df = pd.DataFrame(data)
df.to_csv('dados_consumo_escritorio.csv', index=False)

print("Dataset 'dados_consumo_escritorio.csv' gerado com sucesso!")
