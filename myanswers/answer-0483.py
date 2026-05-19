import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def identificar_variables_redundantes(df, umbral):
    df_numerico = df.select_dtypes(include=['number'])

    if df_numerico.empty:
        return []

    X = add_constant(df_numerico)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif_data = vif_data[vif_data['feature'] != 'const']

    redundant_variables = vif_data[vif_data['VIF'] > umbral]['feature'].tolist()

    return redundant_variables