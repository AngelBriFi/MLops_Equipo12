"""Utilidades compartidas: hash de archivos, esquema/metadata y MLflow."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path

import mlflow
import pandas as pd


def file_hash(filepath: str | Path) -> str:
    """SHA-256 de un archivo (identifica de forma única una versión de datos)."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def git_commit() -> str:
    """Commit actual (para trazabilidad); 'unknown' si no hay git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def setup_mlflow(params: dict) -> None:
    """Configura tracking URI y experimento."""
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    # Bajo `mlflow run .` el experimento y el run padre ya vienen fijados por
    # variables de entorno; no se deben sobreescribir.
    if not os.getenv("MLFLOW_RUN_ID"):
        mlflow.set_experiment(params["mlflow"]["experiment"])


@contextmanager
def tracked_run(run_name: str):
    """Abre un run de MLflow que funciona igual con `make` y con `mlflow run .`.

    Bajo `mlflow run .` todas las etapas comparten el mismo run del proyecto
    (variable MLFLOW_RUN_ID); si cada una registrara en él directamente, los
    parámetros chocarían. Por eso cada etapa se abre como run hijo.
    """
    # pop: si la variable sigue puesta, MLflow volvería a reanudar el run del
    # proyecto también para el run hijo en vez de crear uno nuevo.
    env_run = os.environ.pop("MLFLOW_RUN_ID", None) if mlflow.active_run() is None else None
    if env_run:
        with mlflow.start_run(run_id=env_run):
            with mlflow.start_run(run_name=run_name, nested=True) as run:
                yield run
    else:
        with mlflow.start_run(run_name=run_name) as run:
            yield run


def write_schema_and_metadata(df: pd.DataFrame, parquet_file: Path) -> dict:
    """Escribe schema.json y metadata.json junto al parquet y devuelve metadata."""
    out_dir = parquet_file.parent
    schema_file = out_dir / f"{parquet_file.stem}.schema.json"
    meta_file = out_dir / f"{parquet_file.stem}.metadata.json"
    with open(schema_file, "w", encoding="utf-8") as fh:
        json.dump(pd.io.json.build_table_schema(df), fh, indent=2)
    metadata = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "dataset_hash": file_hash(parquet_file),
    }
    with open(meta_file, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata


def log_dataset_run(df: pd.DataFrame, parquet_file: Path, params: dict,
                    run_name: str) -> None:
    """Registra en MLflow el versionado de un dataset (hash, esquema, nulos)."""
    metadata = write_schema_and_metadata(df, parquet_file)
    setup_mlflow(params)
    with tracked_run(run_name):
        mlflow.log_param("dataset_hash", metadata["dataset_hash"])
        mlflow.log_param("git_commit", git_commit())
        mlflow.log_metric("rows", metadata["rows"])
        mlflow.log_metric("columns", metadata["columns"])
        mlflow.log_metric("null_count", int(df.isna().sum().sum()))
        out_dir = parquet_file.parent
        for suffix in (".schema.json", ".metadata.json"):
            mlflow.log_artifact(str(out_dir / f"{parquet_file.stem}{suffix}"))
