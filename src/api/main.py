from datetime import date, datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import predict_sales
from src.database.db import save_prediction, get_recent_predictions


app = FastAPI(
    title="Retail Demand Forecasting API",
    description="API for predicting retail product demand using a trained LightGBM model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    prediction_date: date = Field(..., example="2017-08-16")
    store_nbr: int = Field(..., example=1)
    family: str = Field(..., example="GROCERY I")
    onpromotion: int = Field(default=0, example=10)


class PredictionResponse(BaseModel):
    prediction_id: int
    prediction_date: date
    store_nbr: int
    family: str
    predicted_sales: float
    model_name: str


@app.get("/")
def root():
    return {
        "message": "Retail Demand Forecasting API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        predicted_sales = predict_sales(
            store_nbr=request.store_nbr,
            family=request.family,
            target_date=str(request.prediction_date),
            onpromotion=request.onpromotion,
            save_to_db=False,
        )

        prediction_id = save_prediction(
            prediction_date=request.prediction_date,
            store_nbr=request.store_nbr,
            family=request.family,
            predicted_sales=predicted_sales,
            model_name="lightgbm_v1",
        )

        return PredictionResponse(
            prediction_id=prediction_id,
            prediction_date=request.prediction_date,
            store_nbr=request.store_nbr,
            family=request.family,
            predicted_sales=round(predicted_sales, 2),
            model_name="lightgbm_v1",
        )

    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/predictions")
def recent_predictions(limit: int = 10):
    try:
        rows = get_recent_predictions(limit=limit)

        results = []

        for row in rows:
            row_dict = row._mapping

            results.append(
                {
                    "id": row_dict["id"],
                    "prediction_date": row_dict["prediction_date"],
                    "store_nbr": row_dict["store_nbr"],
                    "family": row_dict["family"],
                    "predicted_sales": round(row_dict["predicted_sales"], 2),
                    "model_name": row_dict["model_name"],
                    "created_at": row_dict["created_at"],
                }
            )

        return {
            "count": len(results),
            "predictions": results,
        }

    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))