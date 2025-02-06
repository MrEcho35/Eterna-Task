from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[
        logging.FileHandler("app.log"),  
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__)  

app = FastAPI(
    title="Conversion Prediction API",
    description="Bu API, Prophet modeli kullanarak 7 günlük conversion_count tahmini yapar.",
    version="1.0.0"
)

# Load data
def load_data(filepath: str) -> pd.DataFrame:
    """Veriyi JSON dosyasından yükler ve gerekli dönüşümleri yapar."""
    data = pd.read_json(filepath)
    df = pd.DataFrame(data)
    df = df.groupby('date').agg({
        'group_id': 'nunique',  
        'click_count': 'sum',  
        'impression_count': 'sum',  
        'conversion_count': 'sum'   
    }).reset_index()
    return df

df: pd.DataFrame = load_data('data.json')

# Feature Engineering
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Özellik mühendisliği işlemlerini uygular."""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['dayofyear'] = df['date'].dt.dayofyear
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Bayram ve resmi tatil günleri
    holidays: List[str] = ["2024-01-01", "2024-04-23", "2024-05-01", "2024-07-15", "2024-10-29"]
    df["is_holiday"] = df["date"].astype(str).isin(holidays).astype(int)

    # Lag özellikleri
    lags: List[int] = [3, 7, 15]
    for lag in lags:
        df[f"conversion_lag_{lag}"] = df["conversion_count"].shift(lag)

    return df

df: pd.DataFrame = feature_engineering(df)

# Prophet Modeli Eğitme
def train_model(df: pd.DataFrame) -> Prophet:
    """Prophet modelini eğitir ve döndürür."""
    df_prophet = df.rename(columns={"date": "ds", "conversion_count": "y"})
    model = Prophet()
    model.add_regressor('click_count')
    model.add_regressor('month')
    model.add_regressor('impression_count')
    model.add_regressor('weekday')
    model.add_regressor('dayofyear')
    model.add_regressor('is_weekend')
    model.add_regressor('is_holiday')
    logger.info("Prophet modeli eğitiliyor...") 
    model.fit(df_prophet)
    logger.info("Prophet modeli eğitildi.")
    return model

model: Prophet = train_model(df)

# Model Performans Metriği Hesaplama
def calculate_metrics(df: pd.DataFrame, model: Prophet) -> Dict[str, float]:
    """MAPE ve RMSE hesaplar."""
    df = df.rename(columns={"date": "ds", "conversion_count": "y"})  # Prophet formatına dönüştür
    forecast = model.predict(df)
    train = df
    test = df.tail(7)
    actual_values = test['y']
    predicted_values = forecast['yhat'].tail(7)
    rmse: float = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape: float = mean_absolute_percentage_error(actual_values, predicted_values)
    return {"mape": mape, "rmse": rmse}

metrics: Dict[str, float] = calculate_metrics(df, model)

# Pydantic veri modelleri
class PredictionInput(BaseModel):
    days: int = Field(..., gt=0, description="Tahmin yapılacak gün sayısı. 1 veya daha büyük olmalı.")

class PredictionOutput(BaseModel):
    date: str
    conversion_prediction: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
    model_metrics: Dict[str, float]

# API Endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionInput) -> PredictionResponse:

    logger.info(f"Yeni tahmin isteği alındı. Gün sayısı: {data.days}")
    
    future: pd.DataFrame = model.make_future_dataframe(periods=data.days)
    
    last_values = df.iloc[-1]
    future['click_count'] = last_values['click_count']
    future['month'] = last_values['month']
    future['impression_count'] = last_values['impression_count']
    future['weekday'] = last_values['weekday']
    future['dayofyear'] = last_values['dayofyear']
    future['is_weekend'] = last_values['is_weekend']
    future['is_holiday'] = last_values['is_holiday']
    
    forecast: pd.DataFrame = model.predict(future)
    predictions: pd.DataFrame = forecast[['ds', 'yhat']].tail(data.days)

    logger.info("Tahmin işlemi tamamlandı.")

    return PredictionResponse(
        predictions=[
            PredictionOutput(date=str(row.ds.date()), conversion_prediction=float(row.yhat))
            for _, row in predictions.iterrows()
        ],
        model_metrics={
            "mape": float(metrics["mape"]),
            "rmse": float(metrics["rmse"])
        }
    )

# Genel Hata Yönetimi
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Bir hata oluştu: {str(exc)}", exc_info=True)  
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

# Ana Sayfa
@app.get("/", response_class=HTMLResponse)
def read_root() -> str:
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
