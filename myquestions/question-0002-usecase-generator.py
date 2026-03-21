import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generar_caso_de_uso_analizar_operacion_drones():
    # 1. Crear datos aleatorios para el input
    n_rows = np.random.randint(8, 15)
    
    base_date = datetime(2026, 1, 1)
    # Generamos horas de detección aleatorias
    detecciones = [base_date + timedelta(hours=np.random.randint(1, 100)) for _ in range(n_rows)]
    # El límite de autonomía es posterior a la detección
    autonomias = [d + timedelta(hours=np.random.randint(5, 40)) for d in detecciones]
    
    data = {
        'id_evento': range(500, 500 + n_rows),
        'hora_deteccion': [d.strftime('%Y-%m-%d %H:%M:%S') for d in detecciones],
        'limite_autonomia': [a.strftime('%Y-%m-%d %H:%M:%S') for a in autonomias],
        'distancia_objetivo_km': np.random.uniform(2.0, 60.0, n_rows),
        'tipo_alerta': np.random.choice(['Baja', 'Critica', 'Media'], n_rows)
    }
    
    df_input = pd.DataFrame(data)
    # El umbral ahora se llama radio_operativo
    radio_input = float(np.random.randint(10, 25))
    
    # 2. Calcular el output esperado (Misma lógica, nuevos nombres)
    df_expected = df_input.copy()
    df_expected['hora_deteccion'] = pd.to_datetime(df_expected['hora_deteccion'])
    df_expected['limite_autonomia'] = pd.to_datetime(df_expected['limite_autonomia'])
    
    # Diferencia en horas
    diff = df_expected['limite_autonomia'] - df_expected['hora_deteccion']
    df_expected['horas_disponibles'] = diff.dt.total_seconds() / 3600.0
    
    # Filtrado por radio operativo
    df_expected = df_expected[df_expected['distancia_objetivo_km'] > radio_input]
    
    # Codificación (Cambiamos "Express" por "Critica")
    df_expected['es_prioritario'] = (df_expected['tipo_alerta'] == 'Critica').astype(int)
    
    # Selección de columnas finales
    final_output = df_expected[['horas_disponibles', 'distancia_objetivo_km', 'es_prioritario']]
    
    # 3. Estructurar el caso de uso
    input_dict = {
        "df_avistamientos": df_input,
        "radio_operativo": radio_input
    }
    
    return input_dict, final_output

# Ejemplo de ejecución:
if __name__ == "__main__":
    input_data, expected_df = generar_caso_de_uso_analizar_operacion_drones()
    print("INPUT keys:", list(input_data.keys()))
    print("DataFrame sample:")
    print(input_data['df_avistamientos'].head())
    print("Radio de operación:", input_data['radio_operativo'])
    print("\nResultado esperado (Primeras filas):")
    print(expected_df.head())
