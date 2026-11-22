from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd

from src.config import load_params, resolve_path


class PredictRequest(BaseModel):
    lag_1h: float
    lag_2h: float
    lag_6h: float
    lag_12h: float
    lag_24h: float
    lag_7d: float
    media_24h: float
    rolling_max_6h: float
    rolling_max_12h: float
    rolling_max_24h: float
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    mes_sin: float
    mes_cos: float
    Origen_Id: str
    es_fin_semana: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    params_path = resolve_path("params.yaml")
    params = load_params(params_path)

    model_uri = params["mlflow"]["uri"]

    app.state.model_uri = model_uri
    app.state.model = mlflow.pyfunc.load_model(model_uri)

    yield


app = FastAPI(
    title="Mibici Demanda API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    return {
        "status": "ok",
        "model_uri": request.app.state.model_uri,
    }


@app.post("/predict")
def predict(payload: PredictRequest, request: Request):
    try: 
        X = pd.DataFrame([payload.model_dump()])

        float_cols = [
            "lag_1h",
            "lag_2h",
            "lag_6h",
            "lag_12h",
            "lag_24h",
            "lag_7d",
            "media_24h",
            "rolling_max_6h",
            "rolling_max_12h",
            "rolling_max_24h",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "mes_sin",
            "mes_cos",
        ]

        X[float_cols] = X[float_cols].astype("float32")

        X["Origen_Id"] = X["Origen_Id"].astype(str)

        X["es_fin_semana"] = X["es_fin_semana"].astype("int32")


        predictions = request.app.state.model.predict(X)

        return {
            "predictions": predictions.tolist()
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc