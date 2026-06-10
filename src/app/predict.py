import mlflow
import numpy as np

from src.components.model_registry import ModelRegistry

registry = ModelRegistry()
model = registry.load_production_model()


def predict_single(input_data: dict):

    features = np.array([
        input_data["age"],
        input_data["sex"],
        input_data["cp"],
        input_data["trestbps"],
        input_data["chol"],
        input_data["fbs"],
        input_data["restecg"],
        input_data["thalach"],
        input_data["exang"],
        input_data["oldpeak"],
        input_data["slope"],
        input_data["ca"],
        input_data["thal"]
    ]).reshape(1, -1)

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "risk": "HIGH" if prediction == 1 else "LOW"
    }