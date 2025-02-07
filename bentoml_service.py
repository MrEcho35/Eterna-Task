import bentoml
import pandas as pd
from bentoml.io import JSON


# Load model with BentoML
model_ref = bentoml.picklable_model.get("conversion_predictor:latest")
model_runner = model_ref.to_runner()

# Create BentoML service
svc = bentoml.Service("conversion_service", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
async def predict(data: dict):
    df = pd.DataFrame(data)
    forecast = model_runner.predict.run(df)
    return forecast.to_dict(orient="records")

# In order to run
# bentoml serve bentoml_service:svc --port 3000
