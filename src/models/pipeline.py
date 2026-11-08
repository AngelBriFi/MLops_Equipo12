"""Pipelines de Scikit-Learn: preprocesamiento + modelo.

El mismo preprocesamiento se usa para todos los modelos, de modo que la
comparación entre regresión lineal y XGBoost sea justa.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

MODEL_NAMES = ("linear", "xgboost")


def model_columns(params: dict) -> dict[str, list[str]]:
    """Columnas del modelo derivadas de `features` (única fuente de verdad)."""
    f = params["features"]
    numeric = list(f["lags_hours"]) + [f"media_{f['rolling_window_hours']}h"]
    numeric += [f"rolling_max_{w}h" for w in f.get("rolling_max_hours", [])]
    categorical = [f["id_column"]]
    if f.get("cyclical_time", False):
        numeric += ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "mes_sin", "mes_cos"]
    else:
        categorical += ["hora", "dia_semana", "mes"]
    return {"numeric": numeric, "categorical": categorical, "binary": ["es_fin_semana"]}


def feature_columns(params: dict) -> list[str]:
    cols = model_columns(params)
    return cols["numeric"] + cols["categorical"] + cols["binary"]


def build_preprocessor(params: dict) -> ColumnTransformer:
    """Estandariza numéricas, One-Hot a categóricas y deja binarias intactas."""
    cols = model_columns(params)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cols["numeric"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
             cols["categorical"]),
            ("bin", "passthrough", cols["binary"]),
        ],
        sparse_threshold=1.0,   # matriz dispersa: cientos de estaciones one-hot
    )


def resolve_device(requested: str) -> str:
    """'auto' prueba CUDA con un ajuste mínimo y cae a CPU si no hay GPU."""
    if requested != "auto":
        return requested
    try:
        XGBRegressor(device="cuda", tree_method="hist", n_estimators=1).fit(
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
        return "cuda"
    except Exception:
        return "cpu"


def build_regressor(name: str, params: dict):
    seed = params["seed"]
    if name == "linear":
        return LinearRegression()
    if name == "xgboost":
        xgb_cfg = params["training"]["xgboost"]
        return XGBRegressor(
            random_state=seed,
            n_jobs=params["training"]["n_jobs"],
            tree_method=xgb_cfg["tree_method"],
            device=resolve_device(xgb_cfg["device"]),
        )
    raise ValueError(f"Modelo desconocido: {name}. Opciones: {MODEL_NAMES}")


def build_pipeline(name: str, params: dict) -> Pipeline:
    """Pipeline: preprocess -> model."""
    return Pipeline([
        ("preprocess", build_preprocessor(params)),
        ("model", build_regressor(name, params)),
    ])


def search_distributions(params: dict) -> dict:
    """Convierte search_space (YAML) en distribuciones de RandomizedSearchCV."""
    dists = {}
    for key, spec in params["training"]["search_space"].items():
        kind = spec["type"]
        if kind == "int":
            dist = randint(spec["low"], spec["high"] + 1)
        elif kind == "uniform":
            dist = uniform(spec["low"], spec["high"])
        elif kind == "loguniform":
            dist = loguniform(spec["low"], spec["high"])
        elif kind == "choice":
            dist = list(spec["values"])
        else:
            raise ValueError(f"Tipo de distribución desconocido: {kind}")
        dists[f"model__{key}"] = dist
    return dists
