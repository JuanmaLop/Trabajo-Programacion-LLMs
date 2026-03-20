import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

def generar_caso_de_uso_preparar_deteccion_fraude():
    # 1. Crear datos aleatorios para el input
    n_rows = np.random.randint(12, 18)
    tipos = ['Retail', 'Food', 'Tech', 'Services']
    
    data = {
        'id_tx': range(n_rows),
        'monto': [np.random.choice([np.nan, np.random.uniform(10, 5000)]) for _ in range(n_rows)],
        'tipo_comercio': np.random.choice(tipos, n_rows)
    }
    
    df_input = pd.DataFrame(data)
    monto_max_input = float(np.random.randint(2000, 4000))
    
    # 2. Calcular el output esperado (Lógica interna)
    df_step = df_input.copy()
    
    # Imputación de monto
    imputer = SimpleImputer(strategy='median')
    df_step['monto'] = imputer.fit_transform(df_step[['monto']])
    
    # Crear etiqueta antes de escalar (usando el monto imputado)
    y_target = (df_step['monto'] > monto_max_input).astype(int).values
    
    # Codificación de tipo_comercio
    le = LabelEncoder()
    df_step['tipo_comercio'] = le.fit_transform(df_step['tipo_comercio'])
    
    # Escalado de monto
    scaler = StandardScaler()
    df_step['monto'] = scaler.fit_transform(df_step[['monto']])
    
    # Matriz de características X
    X_res = df_step[['tipo_comercio', 'monto']].values
    
    # 3. Estructurar el caso de uso
    input_dict = {
        "df_transacciones": df_input,
        "monto_max_normal": monto_max_input
    }
    
    output = (X_res, y_target)
    
    return input_dict, output

# Ejemplo de uso:
if __name__ == "__main__":
    input_data, expected_output = generar_caso_de_uso_preparar_deteccion_fraude()
    print("INPUT keys:", list(input_data.keys()))
    print("DataFrame sample:")
    print(input_data['df_transacciones'].head())
    print(input_data['df_transacciones'].shape)
    print("monto_max_normal:", input_data['monto_max_normal'])
    print("OUTPUT (numpy.ndarray ['tipo_comercio' , 'monto'])", expected_output[0])
    print("OUTPUT (numpy.ndarray ['sospechoso'])", expected_output[1])