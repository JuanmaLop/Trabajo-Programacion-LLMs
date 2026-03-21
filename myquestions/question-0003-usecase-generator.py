import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def generar_caso_de_uso_preparar_analisis_satelital():
    # --- 1. Parámetros aleatorios ---
    n_rows = random.randint(8, 15)
    n_sensores = random.randint(2, 4)

    # --- 2. Crear DataFrame de Telemetría ---
    sensor_cols = [f"sensor_telemetria_{i}" for i in range(n_sensores)]
    data_sensores = np.random.randn(n_rows, n_sensores) * (np.random.randint(5, 15))
    df = pd.DataFrame(data_sensores, columns=sensor_cols)

    # Introducir fallos de sensores (NaNs)
    nan_mask = np.random.choice([True, False], size=df.shape, p=[0.2, 0.8])
    df[nan_mask] = np.nan

    # --- 3. Columna de Momentos de Captura ---
    start = pd.Timestamp('2024-01-01')
    random_days = np.random.randint(0, 500, size=n_rows)
    momentos = [start + pd.Timedelta(int(d), unit='D') for d in random_days]

    col_momento = 'momento_captura'
    df[col_momento] = [m.strftime("%Y/%m/%d") for m in momentos]

    # Simular errores de transmisión en fechas
    for i in random.sample(range(n_rows), k=1):
        df.at[i, col_momento] = "ERROR_DATA"

    # --- 4. INPUT ---
    input_data = {
        'df_telemetria': df.copy(),
        'col_momento': col_momento
    }

    # --- 5. CÁLCULO DE OUTPUT ESPERADO ---
    df_work = df.copy()

    # A) Fechas
    df_work[col_momento] = pd.to_datetime(df_work[col_momento], errors='coerce', format='%Y/%m/%d')
    if df_work[col_momento].notna().sum() > 0:
        median_date = df_work[col_momento].dropna().median()
    else:
        median_date = pd.Timestamp('2020-01-01')
    df_work[col_momento] = df_work[col_momento].fillna(median_date)

    # B) Features temporales
    df_work['dia_mision'] = df_work[col_momento].dt.day.astype(int)
    df_work['mes_operativo'] = df_work[col_momento].dt.month.astype(int)
    df_work['ciclo_semanal'] = df_work[col_momento].dt.weekday.astype(int)
    df_work['prioridad_fin_semana'] = df_work['ciclo_semanal'].isin([5,6]).astype(int)

    # C) OneHot de ciclo_semanal
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    semana_ohe = ohe.fit_transform(df_work[['ciclo_semanal']])

    # D) Sensores originales (Imputación + Escalado)
    imputer = SimpleImputer(strategy='mean')
    X_num_imputed = imputer.fit_transform(df_work[sensor_cols])
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    # E) Combinación final
    temporal_simple = df_work[['dia_mision', 'mes_operativo', 'prioridad_fin_semana']].to_numpy(dtype=float)
    X_final = np.hstack([X_num_scaled, temporal_simple, semana_ohe])

    return input_data, X_final

# Ejemplo de prueba rápida:
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_preparar_analisis_satelital()
    print("INPUT keys:", list(entrada.keys()))
    print("DataFrame sample:")
    print(entrada['df_telemetria'].head())
    print(entrada['df_telemetria'].shape)
    print("col_momento:", entrada['col_momento'])
    print("OUTPUT shape (features procesadas):", salida_esperada)
