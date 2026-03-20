import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_caso_de_uso_extraer_metricas_logistica():
    # 1. Crear datos aleatorios para el input
    n_rows = np.random.randint(8, 15)
    
    base_date = datetime(2026, 1, 1)
    fechas_registro = [base_date + timedelta(hours=np.random.randint(1, 100)) for _ in range(n_rows)]
    # La fecha límite es siempre posterior a la de registro
    fechas_limite = [fr + timedelta(hours=np.random.randint(10, 50)) for fr in fechas_registro]
    
    data = {
        'id_pedido': range(100, 100 + n_rows),
        'fecha_registro': [f.strftime('%Y-%m-%d %H:%M:%S') for f in fechas_registro],
        'fecha_limite': [f.strftime('%Y-%m-%d %H:%M:%S') for f in fechas_limite],
        'distancia_km': np.random.uniform(1.0, 50.0, n_rows),
        'tipo_servicio': np.random.choice(['Standard', 'Express', 'Eco'], n_rows)
    }
    
    df_input = pd.DataFrame(data)
    umbral_input = float(np.random.randint(5, 15))
    
    # 2. Calcular el output esperado (Lógica interna)
    df_expected = df_input.copy()
    df_expected['fecha_registro'] = pd.to_datetime(df_expected['fecha_registro'])
    df_expected['fecha_limite'] = pd.to_datetime(df_expected['fecha_limite'])
    
    # Diferencia en horas como float
    diff = df_expected['fecha_limite'] - df_expected['fecha_registro']
    df_expected['horas_disponibles'] = diff.dt.total_seconds() / 3600.0
    
    # Filtrado
    df_expected = df_expected[df_expected['distancia_km'] > umbral_input]
    
    # Codificación
    df_expected['es_prioritario'] = (df_expected['tipo_servicio'] == 'Express').astype(int)
    
    # Selección de columnas
    final_output = df_expected[['horas_disponibles', 'distancia_km', 'es_prioritario']]
    
    # 3. Estructurar el caso de uso
    input_dict = {
        "df_pedidos": df_input,
        "umbral_distancia": umbral_input
    }
    
    return input_dict, final_output

# Ejemplo de uso:
input_data, expected_df = generar_caso_de_uso_extraer_metricas_logistica()

if __name__ == "__main__":
    input_data, expected_df = generar_caso_de_uso_extraer_metricas_logistica()
    print("INPUT keys:", list(input_data.keys()))
    print("DataFrame sample:")
    print(input_data['df_pedidos'].head())
    print(input_data['df_pedidos'].shape)
    print("umbral_distancia:", input_data['umbral_distancia'])
    print("OUTPUT ['horas_disponibles', 'distancia_km', 'es_prioritario']:", expected_df)