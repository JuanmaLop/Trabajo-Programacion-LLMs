import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def predecir_emisiones_co2(X_train: pd.DataFrame, y_train: [pd.Series, np.ndarray], X_test: pd.DataFrame) -> np.ndarray:
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return predictions