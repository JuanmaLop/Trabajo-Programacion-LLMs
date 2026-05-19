import pandas as pd
import numpy as np

def calcular_perfil_estadistico(df: pd.DataFrame, grupo_col: str, valor_col: str) -> pd.DataFrame:
    resumen = df.groupby(grupo_col)[valor_col].agg(
        media='mean',
        mediana='median',
        desviacion_std='std',
        minimo='min',
        maximo='max'
    )

    resumen['media'] = resumen['media'].round(2)
    resumen['mediana'] = resumen['mediana'].round(2)
    resumen['desviacion_std'] = resumen['desviacion_std'].round(2)

    resumen = resumen.sort_index()

    return resumen