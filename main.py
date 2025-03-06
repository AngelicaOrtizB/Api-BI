from typing import Optional
from fastapi import FastAPI
import pandas as pd
from joblib import load
from DataModel import DataModel  # Ensure this file exists and is correct

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    # Convert DataModel to DataFrame
    df = pd.DataFrame([dataModel.dict()])  # Ensure correct format
    print("Received Data:", df)  # Debugging step

    # Load trained model
    model = load("modelotelescope2.joblib")

    # Make prediction
    result = model.predict(df)

    return {"prediction": result.tolist()}  # Convert to JSON serializable format
