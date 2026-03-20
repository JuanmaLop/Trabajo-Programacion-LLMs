import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def generar_caso_de_uso_procesar_calidad_cafe():
    # 1. Crear datos aleatorios para el input
    n_rows = np.random.randint(10, 20)
    data = {
        'id_lote': range(n_rows),
        'humedad': np.random.uniform(8.0, 14.0, n_rows),
        'altitud': [np.random.choice([np.nan, np.random.uniform(1000, 2500)]) for _ in range(n_rows)],
        'densidad': np.random.uniform(0.6, 0.9, n_rows),
        'puntuacion': np.random.uniform(70, 95, n_rows)
    }
    df_input = pd.DataFrame(data)
    h_min_input = round(np.random.uniform(9.0, 11.0), 1)

    # 2. Calcular el output esperado (Lógica interna)
    # Filtrado
    df_step = df_input[df_input['humedad'] >= h_min_input].copy()

    # Imputación
    imputer = SimpleImputer(strategy='median')
    df_step['altitud'] = imputer.fit_transform(df_step[['altitud']])

    # Escalado
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_step[['altitud', 'densidad']])

    y_target = df_step['puntuacion']

    # 3. Estructurar el caso de uso
    input_dict = {
        "df_muestras": df_input,
        "humedad_min": h_min_input
    }

    output = (X_scaled, y_target)

    return input_dict, output

# Ejemplo de ejecución del generador:
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_procesar_calidad_cafe()
    print("INPUT keys:", list(entrada.keys()))
    print("DataFrame sample:")
    print(entrada['df_muestras'].head())
    print("humedad_min:", entrada['humedad_min'])
    print("OUTPUT (numpy.ndarray procesadas):", salida_esperada[0])
    print("OUTPUT (panda.Series procesadas):", salida_esperada[1])
    
    
    
    