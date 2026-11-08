"""Carga de configuración central (params.yaml) y resolución de rutas."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = PROJECT_ROOT / "params.yaml"


def load_params(path: str | Path | None = None) -> dict:
    """Lee params.yaml. El URI de MLflow puede sobreescribirse por entorno."""
    params_path = Path(path) if path else DEFAULT_PARAMS
    with open(params_path, encoding="utf-8") as fh:
        params = yaml.safe_load(fh)
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        params["mlflow"]["tracking_uri"] = env_uri
    return params


def resolve_path(relative: str | Path) -> Path:
    """Convierte una ruta relativa al proyecto en absoluta."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
