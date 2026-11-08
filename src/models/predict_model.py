# -*- coding: utf-8 -*-
"""Inferencia con el modelo champion (Model Registry o archivo local)."""
from __future__ import annotations

import logging

import click
import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.config import load_params, resolve_path
from src.models.pipeline import feature_columns


def load_model(params: dict, model_uri: str | None = None):
    """Carga desde el Registry (p. ej. models:/mibici-demand-xgboost@champion)
    o, si no se indica URI, desde models/champion.joblib."""
    if model_uri:
        mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
        return mlflow.sklearn.load_model(model_uri)
    return joblib.load(resolve_path(params["paths"]["models"]) / "champion.joblib")


def predict(model, df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Devuelve estación, hora y demanda pronosticada (>= 0)."""
    f = params["features"]
    out = df[[f["id_column"], f["time_column"]]].copy()
    out["demanda_pronosticada"] = np.clip(model.predict(df[feature_columns(params)]), 0, None)
    return out


@click.command()
@click.option("--params", "params_path", default=None)
@click.option("--input", "input_file", default=None, help="Parquet de features.")
@click.option("--output", "output_file", default="reports/predictions.parquet")
@click.option("--model-uri", default=None)
def main(params_path, input_file, output_file, model_uri):
    params = load_params(params_path)
    input_file = resolve_path(input_file) if input_file else (
        resolve_path(params["paths"]["processed"]) / params["features"]["processed_file"])
    model = load_model(params, model_uri)
    preds = predict(model, pd.read_parquet(input_file), params)
    out = resolve_path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    logging.getLogger(__name__).info("%d predicciones -> %s", len(preds), out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv(find_dotenv())
    main()
