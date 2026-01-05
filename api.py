from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from infare import predict_

app =FastAPI(title="Titanic")

class features(BaseModel):
    features : List[float]

@app.post("/predict")
async def predict(features: features):
        prediction = predict_(features.features)
        return {"prediction": prediction}

