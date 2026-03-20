import pandas as pd
import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def generar_caso_de_uso_preparar_features_temporales():
    """
    Genera un caso de uso aleatorio (input y output esperado)
    para la función preparar_features_temporales(df, fecha_col).

    Output:
      input_data: dict con keys {'df', 'fecha_col'}
      output_data: np.ndarray (matriz de features procesadas)
    """
    # --- 1. Parámetros aleatorios de tamaño ---
    n_rows = random.randint(6, 18)      # filas
    n_numeric = random.randint(1, 4)    # columnas numéricas además de la fecha

    # --- 2. Crear DataFrame base con columnas numéricas ---
    numeric_cols = [f"num_{i}" for i in range(n_numeric)]
    data_num = np.random.randn(n_rows, n_numeric) * (np.random.randint(1,5))  # variación de escala
    df = pd.DataFrame(data_num, columns=numeric_cols)

    # Introducir NaNs aleatorios en ~10-25% de las celdas numéricas
    nan_mask = np.random.choice([True, False], size=df.shape, p=[0.15, 0.85])
    df[nan_mask] = np.nan

    # --- 3. Crear columna de fechas con algunos NaNs ---
    start = pd.Timestamp('2018-01-01')
    # generar fechas dentro de ~3 años
    random_days = np.random.randint(0, 365*3, size=n_rows)
    fechas = [start + pd.Timedelta(int(d), unit='D') for d in random_days]
    # Convertir a strings (simulando entrada real)
    fecha_col = 'fecha_evento'
    df[fecha_col] = [d.strftime("%Y-%m-%d") for d in fechas]

    # introducir NaNs en fechas (aprox 10%)
    for i in random.sample(range(n_rows), k=max(1, n_rows // 10)):
        df.at[i, fecha_col] = np.nan

    # --- 4. Construcción del INPUT (pasamos copia) ---
    input_data = {
        'df': df.copy(),
        'fecha_col': fecha_col
    }

    # --- 5. Calcular OUTPUT esperado replicando la lógica ---
    df_work = df.copy()

    # A) parsear fechas y rellenar NaT con mediana (o valor por defecto)
    df_work[fecha_col] = pd.to_datetime(df_work[fecha_col], errors='coerce')
    if df_work[fecha_col].notna().sum() > 0:
        median_date = df_work[fecha_col].dropna().median()
    else:
        median_date = pd.Timestamp('2020-01-01')
    df_work[fecha_col] = df_work[fecha_col].fillna(median_date)

    # B) crear columnas temporales
    df_work['day'] = df_work[fecha_col].dt.day.astype(int)
    df_work['month'] = df_work[fecha_col].dt.month.astype(int)
    df_work['weekday'] = df_work[fecha_col].dt.weekday.astype(int)  # 0..6
    df_work['is_weekend'] = df_work['weekday'].isin([5,6]).astype(int)

    # C) OneHot encode weekday (7 columnas)
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    weekday_ohe = ohe.fit_transform(df_work[['weekday']])

    # D) Seleccionar columnas numéricas originales y aplicar imputación + escalado
    numeric_original = numeric_cols  # lista
    imputer = SimpleImputer(strategy='mean')
    X_num_imputed = imputer.fit_transform(df_work[numeric_original])

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num_imputed)

    # E) Construir la matriz final: [num_scaled] + [day, month, is_weekend] + [weekday_ohe]
    temporal_simple = df_work[['day', 'month', 'is_weekend']].to_numpy(dtype=float)
    X_final = np.hstack([X_num_scaled, temporal_simple, weekday_ohe])

    output_data = X_final

    return input_data, output_data

# --- Ejemplo de uso rápido ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_preparar_features_temporales()
    print("INPUT keys:", list(entrada.keys()))
    print("DataFrame sample:")
    print(entrada['df'].head())
    print("OUTPUT shape (features procesadas):", salida_esperada)