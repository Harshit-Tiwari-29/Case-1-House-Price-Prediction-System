import pickle
import numpy as np
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="House Price API")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'linear_regression_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ CRITICAL ERROR: Could not load model from {MODEL_PATH}")
    print(f"Details: {e}")
    model = None

SCALER_MEAN = np.array([
    2402.54766, 3.00046, 2.00282, 1961.42919, 63.57081, # Numeric
    0.25050, 0.24861, 0.25117, 0.24972,               # Locations
    0.23834, 0.35708, 0.40458,                        # Types
    1.38167                                           # Condition
])

SCALER_SCALE = np.array([
    923.41668, 1.41509, 0.81661, 35.75256, 35.75256,  # Numeric
    0.43330, 0.43221, 0.43369, 0.43285,               # Locations
    0.42607, 0.47914, 0.49081,                        # Types
    0.89902                                           # Condition
])

class HouseInput(BaseModel):
    size_sqft: float
    bedrooms: int
    bathrooms: float
    year_built: int
    location: str
    property_type: str
    condition: str

@app.post("/predict")
def predict_price(data: HouseInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not found. Check terminal for path error.")

    try:
        property_age = 2025 - data.year_built
        
        cond_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'New': 3}
        cond_encoded = cond_map.get(data.condition, 1)

        raw_features = np.array([
            data.size_sqft,
            data.bedrooms,
            data.bathrooms,
            data.year_built,
            property_age,
            # Location OHE (CityA, CityB, CityC, CityD)
            1 if data.location == 'CityA' else 0,
            1 if data.location == 'CityB' else 0,
            1 if data.location == 'CityC' else 0,
            1 if data.location == 'CityD' else 0,
            # Type OHE (Condo, Single, Townhouse)
            1 if data.property_type == 'Condominium' else 0,
            1 if data.property_type == 'Single Family' else 0,
            1 if data.property_type == 'Townhouse' else 0,
            cond_encoded
        ])

        scaled_features = (raw_features - SCALER_MEAN) / SCALER_SCALE

        prediction = model.predict(scaled_features.reshape(1, -1))
        
        return {"predicted_price": round(float(prediction[0]), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Calculation Error: {str(e)}")