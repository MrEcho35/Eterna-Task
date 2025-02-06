import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

def detect_drift(df: pd.DataFrame, model: Prophet, days: int, threshold: float) -> dict:
    """
    Detects if there is a model drift or not.
    
    - `days`: How many days of data will be used?
    - `threshold`: By what percentage does RMSE or MAPE increase, drift will be detected?
    """
    
    df = df.rename(columns={"date": "ds", "conversion_count": "y"})  # Prophet formatına çevir
    
    # Eğer yeterli veri yoksa hata döndür
    if len(df) < days:
        return {"error": f"Not enough data! At least {days} amount of data is required."}
    
    # Take last 'days' amount of data
    train = df[:-days]
    test = df[-days:]

    # Predict
    forecast = model.predict(test)
    
    # Take actual and predicted values
    actual_values = test['y'].values
    predicted_values = forecast['yhat'].values

    # Metric calculations
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape = mean_absolute_percentage_error(actual_values, predicted_values)

    # Drift detection
    drift_detected = bool(rmse > threshold * rmse or mape > threshold * mape)

    return {"drift_detected": drift_detected, "rmse": rmse, "mape": mape}
