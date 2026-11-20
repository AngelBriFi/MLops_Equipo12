from contextlib import asynccontextmanager

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.config import load_params, resolve_path


class PredictRequest(BaseModel):
    instances: list[list[float]]


@asynccontextmanager
async def lifespan(app: FastAPI):

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
        predictions = request.app.state.model.predict(
            payload.instances
        )

        return {
            "model_uri": request.app.state.model_uri,
            "predictions": predictions.tolist(),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc