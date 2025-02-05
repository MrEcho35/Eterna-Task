from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

app = FastAPI(
    title="Conversion Prediction API",
    description="Bu API, Prophet modeli kullanarak 7 günlük conversion_count tahmini yapar.",
    version="1.0.0"
)

# Load data
data = pd.read_json('data.json')
df = pd.DataFrame(data)
df = df.groupby('date').agg({
    'group_id': 'nunique',  
    'click_count': 'sum',  
    'impression_count': 'sum',  
    'conversion_count': 'sum'   
}).reset_index()

# Feature Engineering

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['dayofyear'] = df['date'].dt.dayofyear
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# Turkish Holidays
holidays = ["2024-01-01", "2024-04-23", "2024-05-01", "2024-07-15", "2024-10-29"]
df["is_holiday"] = df["date"].astype(str).isin(holidays).astype(int)

# Lags
lags = [3, 7, 15]
for lag in lags:
    df[f"conversion_lag_{lag}"] = df["conversion_count"].shift(lag)



# Train Prophet Model
df_prophet = df.rename(columns={"date": "ds", "conversion_count": "y"})
model = Prophet()

model.add_regressor('click_count')
model.add_regressor('impression_count')
model.add_regressor('weekday')
model.add_regressor('is_weekend')
model.add_regressor('is_holiday')

model.fit(df_prophet)

forecast = model.predict(df_prophet)
train = df_prophet
test = df_prophet.tail(7)  # For example, we can use the last 7 data points for testing

# The actual values for the last 7 days in the test set
actual_values = test['y']

# The predicted values from the Prophet model (forecasted)
predicted_values = forecast['yhat'].tail(7)

# Calculate MAPE and RMSE
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

mape = mean_absolute_percentage_error(actual_values,predicted_values)


# Pydantic data handling
class PredictionInput(BaseModel):
    days: int = Field(..., gt=0, description="Tahmin yapılacak gün sayısı. 1 veya daha büyük olmalı.")

class PredictionOutput(BaseModel):
    date: str
    conversion_prediction: float

class PredictionResponse(BaseModel):
    predictions: list[PredictionOutput]
    model_metrics: dict

# API Endpoint for prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput):
    future = model.make_future_dataframe(periods=data.days)

    future['click_count'] = df['click_count'].iloc[-1]  # Using the last available click_count value
    future['impression_count'] = df['impression_count'].iloc[-1]  # Using the last available impression_count value
    future['weekday'] = df['weekday'].iloc[-1]
    future['is_weekend'] = df['is_weekend'].iloc[-1]
    future['is_holiday'] = df['is_holiday'].iloc[-1]

    forecast = model.predict(future)

    # Predict the amount of days requested
    predictions = forecast[['ds', 'yhat']].tail(data.days)

    return PredictionResponse(
        predictions=[
            PredictionOutput(date=str(row.ds.date()), conversion_prediction=float(row.yhat))
            for _, row in predictions.iterrows()
        ],
        model_metrics={
            "mape": float(mape) if mape is not None else "N/A",
            "rmse": float(rmse)
        }
    )

# Exception Handling
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

# Endpoint for Main Page
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head>
            <title>Conversion Prediction API</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin: 50px;">
            <h1>🚀 Conversion Prediction API'ye Hoş Geldiniz!</h1>
            <p>Bu API, Prophet kullanarak conversion_count tahmini yapar.</p>
            <p><a href='/docs'>📖 API Dokümantasyonu</a></p>
            <p><a href='/redoc'>📘 ReDoc Dokümantasyonu</a></p>
        </body>
    </html>
    """

# To run the app write the command down below to the shell:
# uvicorn basic_api:app --reload
