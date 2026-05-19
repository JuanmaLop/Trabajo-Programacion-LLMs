import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def detectar_sesgo_creditos(df: pd.DataFrame):
    X = df.drop(columns=['aprobado'])
    y = df['aprobado']

    model = LogisticRegression(random_state=42, solver='liblinear')
    # Entrenar el modelo antes de hacer predicciones
    model.fit(X, y)

    df['prediccion_aprobado'] = model.predict(X)

    reporte = {}
    generos = df['genero'].unique()
    generos.sort()

    for g in generos:
        subset = df[df['genero'] == g]
        actual_aprobado = subset['aprobado']
        prediccion_aprobado = subset['prediccion_aprobado']

        n_solicitantes = len(subset)
        tasa_aprobacion_real = actual_aprobado.mean()
        tasa_aprobacion_predicha = prediccion_aprobado.mean()
        accuracy = accuracy_score(actual_aprobado, prediccion_aprobado)
        precision = precision_score(actual_aprobado, prediccion_aprobado, zero_division=0)
        recall = recall_score(actual_aprobado, prediccion_aprobado, zero_division=0)
        f1 = f1_score(actual_aprobado, prediccion_aprobado, zero_division=0)

        reporte[f'genero_{g}'] = {
            'n_solicitantes': n_solicitantes,
            'tasa_aprobacion_real': tasa_aprobacion_real,
            'tasa_aprobacion_predicha': tasa_aprobacion_predicha,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    if len(generos) == 2:
        g0_actual_rate = reporte[f'genero_{generos[0]}']['tasa_aprobacion_real']
        g1_actual_rate = reporte[f'genero_{generos[1]}']['tasa_aprobacion_real']
        diff_actual = abs(g0_actual_rate - g1_actual_rate)

        g0_pred_rate = reporte[f'genero_{generos[0]}']['tasa_aprobacion_predicha']
        g1_pred_rate = reporte[f'genero_{generos[1]}']['tasa_aprobacion_predicha']
        diff_pred = abs(g0_pred_rate - g1_pred_rate)

        reporte['diferencia_tasa_aprobacion_real'] = diff_actual
        reporte['diferencia_tasa_aprobacion_predicha'] = diff_pred
        reporte['sesgo_detectado_real'] = diff_actual > 0.05
        reporte['sesgo_detectado_predicho'] = diff_pred > 0.05

    return reporte
