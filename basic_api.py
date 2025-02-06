from fastapi import FastAPI, HTTPException, Request, APIRouter, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from typing import List, Dict
import redis
import json
import logging
import threading
from drift_detection import detect_drift
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
    description="This API makes 7 day conversion_count forecasting using Prophet model.",
    version="1.0.0"
)

# Rate Limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Too many requests, please try again later."}
    )

# Global Rate Limit
@app.middleware("http")
async def rate_limiter_middleware(request: Request, call_next):
    response = await limiter.limit("50/minute")(call_next)(request)
    return response

# Load data
def load_data(filepath: str) -> pd.DataFrame:
    """Loads data from JSON and makes the necessary groupby operation"""
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
    """Makes feature engineering."""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['weekday'] = df['date'].dt.weekday
    df['dayofyear'] = df['date'].dt.dayofyear
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Turkish Holidays
    holidays: List[str] = ["2024-01-01", "2024-04-23", "2024-05-01", "2024-07-15", "2024-10-29"]
    df["is_holiday"] = df["date"].astype(str).isin(holidays).astype(int)

    # Lags
    lags: List[int] = [3, 7, 15]
    for lag in lags:
        df[f"conversion_lag_{lag}"] = df["conversion_count"].shift(lag)

    return df

df: pd.DataFrame = feature_engineering(df)

# Prophet Model Training
def train_model(df: pd.DataFrame) -> Prophet:
    """Trains and returns the Prophet Model."""
    df_prophet = df.rename(columns={"date": "ds", "conversion_count": "y"})
    best_params = {'changepoint_prior_scale': 0.01,
                    'growth': 'linear',
                    'holidays_prior_scale': 1,
                    'seasonality_mode': 'additive',
                    'seasonality_prior_scale': 10}
    
    model = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                    seasonality_prior_scale=best_params['seasonality_prior_scale'],
                    holidays_prior_scale=best_params['holidays_prior_scale'],
                    seasonality_mode=best_params['seasonality_mode'],
                    growth=best_params['growth']
                    ) # Optimized parameters
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
global model  
model: Prophet = train_model(df)

# Metric calculation
def calculate_metrics(df: pd.DataFrame, model: Prophet) -> Dict[str, float]:
    """Calculates MAPE and RMSE."""
    df = df.rename(columns={"date": "ds", "conversion_count": "y"})  # Turns them into Prophet format
    forecast = model.predict(df)
    train = df
    test = df.tail(7) # Last 7 days are used for validation since future 7 days are meant to be predicted.
    actual_values = test['y']
    predicted_values = forecast['yhat'].tail(7)
    rmse: float = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape: float = mean_absolute_percentage_error(actual_values, predicted_values)
    return {"mape": mape, "rmse": rmse}

metrics: Dict[str, float] = calculate_metrics(df, model)

# Pydantic data modeling
class PredictionInput(BaseModel):
    days: int = Field(..., gt=0, description="Days to be predicted should be bigger than or equal to 1.")

class PredictionOutput(BaseModel):
    date: str
    conversion_prediction: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
    model_metrics: Dict[str, float]

class DriftCheckInput(BaseModel):
    days: int = Field(30, gt=0, description="How many days of data will be used to check drift? Default: 30")
    threshold: float = Field(0.2, gt=0, description="Drift detection threshold. Default: 0.2")


def get_redis():
    return redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_cached_prediction(redis_client, input_data):
    key = f"prediction:{json.dumps(input_data, sort_keys=True)}"
    cached_value = redis_client.get(key)
    if cached_value:
        return json.loads(cached_value)
    return None

def set_cached_prediction(redis_client, input_data, result, expire_time=300):
    key = f"prediction:{json.dumps(input_data, sort_keys=True)}"
    redis_client.set(key, json.dumps(result), ex=expire_time)

# API Endpoint
@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("10/minute")
def predict(request: Request, data: PredictionInput, redis_client=Depends(get_redis)) -> PredictionResponse:
    logger.info(f"New prediction request is taken. Days: {data.days}")

    input_data = {"days": data.days}
    cached_result = get_cached_prediction(redis_client, input_data)
    if cached_result:
        logger.info("Returning cached prediction.")
        return cached_result

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

    response = PredictionResponse(
        predictions=[
            PredictionOutput(date=str(row.ds.date()), conversion_prediction=float(row.yhat))
            for _, row in predictions.iterrows()
        ],
        model_metrics={
            "mape": float(metrics["mape"]),
            "rmse": float(metrics["rmse"])
        }
    )

    # Sonucu Redis cache'e ekle
    set_cached_prediction(redis_client, input_data, response.dict())

    logger.info("Prediction complete.")
    return response

# General Exception Handler
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"An unexpected error occured: {str(exc)}", exc_info=True)  
    return JSONResponse(
        status_code=500,
        content={"error": f"An unexpected error occurred: {str(exc)}"}
    )

router = APIRouter()


def retrain_model():
    """Modeli yeniden eğitir ve günceller."""
    global model  # Güncellenen modeli tüm sistemde kullanmak için
    logger.info("Model retraining...")
    model = train_model(df)  # Train new model
    logger.info("New model added to the system!")


@router.post("/drift-check")
@limiter.limit("5/minute")
def check_model_drift(request: Request ,data: DriftCheckInput, background_tasks: BackgroundTasks):
    """
    Checks if there is a drift.
    If there is a drift, it adds a task to retrain it.
    """
    drift_results = detect_drift(df, model, data.days, data.threshold)
    
    if drift_results.get("error"):
        return {"error": drift_results["error"]}
    
    if drift_results["drift_detected"]:
        background_tasks.add_task(retrain_model)  # Retrain model in the background
    
    return drift_results

@router.post("/update-model")
def update_model():
    """
    Adds model manually to the system and retrains it.
    """
    threading.Thread(target=retrain_model).start()
    return {"message": "Model retraining begin."}

app.include_router(router)

# Main Page
@app.get("/", response_class=HTMLResponse)
def read_root() -> str:
    return """
    <html>
        <head>
            <title>Conversion Prediction API</title>
        </head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin: 50px;">
            <h1>🚀 Welcome to Conversion Prediction API!</h1>
            <p>This API makes conversion_count forecasting using Prophet model.</p>
            <p><a href='/docs'>📖 API Documentation</a></p>
            <p><a href='/redoc'>📘 ReDoc Documentation</a></p>
        </body>
    </html>
    """

# To run the app write the command down below to the shell:
# uvicorn basic_api:app --reload
