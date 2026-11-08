# -*- coding: utf-8 -*-
"""Ingesta y limpieza: CSV mensuales de MiBici -> parquet 'interim'.

Rol principal: Data Engineer.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.config import load_params, resolve_path
from src.data.validate_dataset import validate_dataset
from src.utils.helpers import log_dataset_run

logger = logging.getLogger(__name__)


def list_raw_files(raw_dir: Path, pattern: str) -> list[Path]:
    """Busca recursivamente los CSV de datos abiertos (ordenados)."""
    files = sorted(p for p in raw_dir.rglob("*.csv") if pattern in p.name)
    if not files:
        raise FileNotFoundError(
            f"No se encontraron CSV con '{pattern}' en {raw_dir}")
    return files


def validate_schema(df: pd.DataFrame, required: list[str], source: str) -> None:
    """Falla temprano si a un archivo le falta alguna columna requerida."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source}: faltan columnas {missing}")


def read_raw_csvs(files: list[Path], keep: list[str], encoding: str) -> pd.DataFrame:
    """Lee solo las columnas necesarias de cada archivo y las unifica.

    Al leer únicamente `keep` se evita la columna de año de nacimiento, que
    llegó corrupta en julio y con dos nombres distintos entre archivos.
    """
    frames = []
    for f in files:
        part = pd.read_csv(f, encoding=encoding,
                           usecols=lambda c: c in keep)
        validate_schema(part, keep, f.name)
        frames.append(part[keep])
    return pd.concat(frames, ignore_index=True)


def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina nulos y duplicados y corrige tipos."""
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    for col in ("Origen_Id", "Destino_Id"):
        df[col] = df[col].astype("int64").astype("str")
    for col in ("Inicio_del_viaje", "Fin_del_viaje"):
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
    return df


def build_interim_dataset(params: dict, raw_dir: Path | None = None,
                          out_dir: Path | None = None) -> Path:
    """Pipeline completo de ingesta; devuelve la ruta del parquet generado."""
    data_cfg = params["data"]
    raw_dir = raw_dir or resolve_path(params["paths"]["raw"])
    out_dir = out_dir or resolve_path(params["paths"]["interim"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_raw_files(raw_dir, data_cfg["raw_file_pattern"])
    logger.info("Leyendo %d archivos", len(files))
    df = clean_trips(read_raw_csvs(files, data_cfg["keep_columns"],
                                   data_cfg["encoding"]))
    validate_dataset(df)   # reglas de consistencia (nulos, fechas, fin >= inicio)
    parquet_file = out_dir / data_cfg["interim_file"]
    df.to_parquet(parquet_file)
    log_dataset_run(df, parquet_file, params, run_name="dataset-interim")
    logger.info("Dataset interim: %d filas -> %s", len(df), parquet_file)
    return parquet_file


@click.command()
@click.option("--params", "params_path", default=None,
              help="Ruta a params.yaml (opcional).")
@click.option("--raw-dir", default=None, help="Sobrescribe paths.raw.")
@click.option("--out-dir", default=None, help="Sobrescribe paths.interim.")
def main(params_path, raw_dir, out_dir):
    """CSV crudos (data/raw) -> parquet limpio (data/interim)."""
    params = load_params(params_path)
    build_interim_dataset(
        params,
        raw_dir=resolve_path(raw_dir) if raw_dir else None,
        out_dir=resolve_path(out_dir) if out_dir else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    load_dotenv(find_dotenv())
    main()
