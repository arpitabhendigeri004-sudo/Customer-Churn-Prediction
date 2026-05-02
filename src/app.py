from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load bundle (model + scaler + threshold)
bundle = joblib.load("models/churn_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]

app = FastAPI()

# Input schema
class Customer(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(customer: Customer):
    data = np.array([list(customer.model_dump().values())])

    # Apply same scaling
    data_scaled = scaler.transform(data)

    # Probability
    prob = model.predict_proba(data_scaled)[0][1]

    # Apply same threshold as training
    prediction = int(prob > threshold)

    return {
        "churn_prediction": prediction,
        "churn_probability": float(prob)
    }