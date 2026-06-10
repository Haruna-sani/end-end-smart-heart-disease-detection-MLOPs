from fastapi import FastAPI
from src.app.schemas import HeartRequest
from src.app.predict import predict_single

app = FastAPI(title="Heart Disease Prediction API")


@app.get("/")
def home():
    return {"message": "Heart Disease API is running"}


@app.post("/predict")
def predict(request: HeartRequest):

    result = predict_single(request.dict())
    return result