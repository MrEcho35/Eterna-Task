import bentoml
from prophet import Prophet
import pandas as pd

# Load and preprocess data
data = pd.read_json('data.json')
df = pd.DataFrame(data)
df = df.groupby('date').agg({
    'group_id': 'nunique',
    'click_count': 'sum',
    'impression_count': 'sum',
    'conversion_count': 'sum'
}).reset_index()

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['weekday'] = df['date'].dt.weekday
df['dayofyear'] = df['date'].dt.dayofyear
df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

# Turkish Holidays
holidays = ["2024-01-01", 
            "2024-04-23", 
            "2024-05-01", 
            "2024-05-19", 
            "2024-07-15", 
            "2024-08-30", 
            "2024-10-29", 
            "2025-01-01"]
df["is_holiday"] = df["date"].astype(str).isin(holidays).astype(int)

# Create lag features
lags = [3, 7, 15]
for lag in lags:
    df[f"conversion_lag_{lag}"] = df["conversion_count"].shift(lag)

# Prepare data for Prophet model
df_prophet = df.rename(columns={"date": "ds", "conversion_count": "y"})
best_params = {
    'changepoint_prior_scale': 0.01,
    'growth': 'linear',
    'holidays_prior_scale': 0.1,
    'seasonality_mode': 'additive',
    'seasonality_prior_scale': 50
}

# Initialize Prophet model
model = Prophet(
    changepoint_prior_scale=best_params['changepoint_prior_scale'],
    seasonality_prior_scale=best_params['seasonality_prior_scale'],
    holidays_prior_scale=best_params['holidays_prior_scale'],
    seasonality_mode=best_params['seasonality_mode'],
    growth=best_params['growth']
)

# Add regressors to the model
model.add_regressor('click_count')
model.add_regressor('month')
model.add_regressor('impression_count')
model.add_regressor('weekday')
model.add_regressor('dayofyear')
model.add_regressor('is_weekend')
model.add_regressor('is_holiday')

# Fit the model
model.fit(df_prophet)

# Save the model with BentoML
saved_model = bentoml.picklable_model.save_model("conversion_predictor", model)

print(f"Model saved successfully: {saved_model}")

# In order to see the list of models:
# bentoml models list
