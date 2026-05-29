import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def detectar_sesgo_creditos(df):
    df = pd.DataFrame(df)
    features_cols = ['edad', 'ingresos', 'puntaje_credito', 'genero']
    X = df[features_cols]
    y = df['aprobado']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y 
    )

    model = RandomForestClassifier(
        n_estimators=200, 
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    X_test_df = pd.DataFrame(X_test, columns=X.columns) 
    predictions = model.predict(X_test_df)

    resultados_test = pd.DataFrame({
        'genero': X_test_df['genero'].astype(int),
        'aprobado_real': y_test,
        'prediccion_aprobado': predictions
    })

    reporte = {}
    generos = sorted(resultados_test['genero'].unique())

    for g in generos:
        subset = resultados_test[resultados_test['genero'] == g]
        actual = subset['aprobado_real']
        pred = subset['prediccion_aprobado']

        reporte[f'genero_{g}'] = {
            'n_solicitantes': int(len(subset)),
            'tasa_aprobacion_real': float(actual.mean()),
            'tasa_aprobacion_predicha': float(pred.mean()),
            'accuracy': float(accuracy_score(actual, pred)),
            'precision': float( precision_score(actual, pred, zero_division=0)),
            'recall': float(recall_score(actual, pred, zero_division=0)),
            'f1_score': float(f1_score(actual, pred, zero_division=0))
        }

    if len(generos) == 2:
        g0_real = reporte[f'genero_{generos[0]}']['tasa_aprobacion_real']
        g1_real = reporte[f'genero_{generos[1]}']['tasa_aprobacion_real']
        diff_real = abs(g0_real - g1_real)
        g0_pred = reporte[f'genero_{generos[0]}']['tasa_aprobacion_predicha']
        g1_pred = reporte[f'genero_{generos[1]}']['tasa_aprobacion_predicha']
        diff_pred = abs(g0_pred - g1_pred)

        reporte['diferencia_tasa_aprobacion_real'] = float(diff_real)
        reporte['diferencia_tasa_aprobacion_predicha'] = float(diff_pred)
        reporte['sesgo_detectado_real'] = bool(diff_real > 0.05)
        reporte['sesgo_detectado_predicho'] = bool(diff_pred > 0.05)

    return reporte
