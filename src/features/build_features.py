# -*- coding: utf-8 -*-
"""Construcción de variables: viajes -> demanda por estación-hora + señales.

Rol principal: Data Scientist.
"""
from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.config import load_params, resolve_path
from src.utils.helpers import log_dataset_run

logger = logging.getLogger(__name__)


def crear_demanda(trips: pd.DataFrame, id_col: str = "Origen_Id",
                  time_col: str = "fecha_hora",
                  target: str = "demanda") -> pd.DataFrame:
    """Cuenta los viajes que inician en cada estación por hora."""
    demanda = (
        trips.groupby([id_col, pd.Grouper(key="Inicio_del_viaje", freq="h")])
        .size().reset_index(name=target)
        .rename(columns={"Inicio_del_viaje": time_col})
    )
    return demanda.sort_values([id_col, time_col]).reset_index(drop=True)


def completar_horas_vacias(demanda: pd.DataFrame, id_col: str = "Origen_Id",
                           time_col: str = "fecha_hora") -> pd.DataFrame:
    """Agrega las horas sin viajes con demanda 0 (malla completa estación x hora)."""
    horas = pd.date_range(demanda[time_col].min().floor("h"),
                          demanda[time_col].max().floor("h"), freq="h")
    malla = pd.MultiIndex.from_product(
        [demanda[id_col].unique(), horas], names=[id_col, time_col])
    return (demanda.set_index([id_col, time_col])
            .reindex(malla, fill_value=0).reset_index())


def agregar_historicos(demanda: pd.DataFrame, lags: dict[str, int],
                       window: int, rolling_max_hours: list[int] | None = None,
                       id_col: str = "Origen_Id", time_col: str = "fecha_hora",
                       target: str = "demanda") -> pd.DataFrame:
    """Rezagos, media móvil y máximos móviles por estación. Solo usa el pasado.

    Las ventanas móviles se calculan *dentro de cada estación* (groupby +
    transform), de modo que una estación nunca ve datos de la anterior.
    """
    df = demanda.sort_values([id_col, time_col]).copy()
    grouped = df.groupby(id_col)[target]
    for name, hours in lags.items():
        df[name] = grouped.shift(hours)
    df[f"media_{window}h"] = grouped.transform(
        lambda s: s.shift(1).rolling(window).mean())
    for w in rolling_max_hours or []:
        df[f"rolling_max_{w}h"] = grouped.transform(
            lambda s, w=w: s.shift(1).rolling(w, min_periods=1).max())
    return df.dropna().reset_index(drop=True)


def agregar_calendario(df: pd.DataFrame, time_col: str = "fecha_hora",
                       cyclical: bool = False) -> pd.DataFrame:
    """Hora, día de la semana (lunes=0), mes y fin de semana.

    Con `cyclical=True` agrega además seno/coseno de hora, día y mes, para que
    el modelo vea que las 23:00 y las 00:00 (o diciembre y enero) son vecinas.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df["hora"] = df[time_col].dt.hour
    df["dia_semana"] = df[time_col].dt.day_of_week
    df["mes"] = df[time_col].dt.month
    df["es_fin_semana"] = (df["dia_semana"] >= 5).astype(int)
    if cyclical:
        for col, period, short in (("hora", 24, "hour"), ("dia_semana", 7, "dow"),
                                   ("mes", 12, "mes")):
            angle = 2 * np.pi * df[col] / period
            df[f"{short}_sin"] = np.sin(angle)
            df[f"{short}_cos"] = np.cos(angle)
    return df


def build_feature_table(trips: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Aplica los pasos en orden usando la configuración central."""
    f = params["features"]
    demanda = crear_demanda(trips, f["id_column"], f["time_column"], f["target"])
    demanda = completar_horas_vacias(demanda, f["id_column"], f["time_column"])
    demanda = agregar_historicos(demanda, f["lags_hours"],
                                 f["rolling_window_hours"],
                                 f.get("rolling_max_hours", []), f["id_column"],
                                 f["time_column"], f["target"])
    return agregar_calendario(demanda, f["time_column"], f.get("cyclical_time", False))


def build_features_dataset(params: dict, in_file: Path | None = None,
                           out_dir: Path | None = None) -> Path:
    in_file = in_file or (resolve_path(params["paths"]["interim"])
                          / params["data"]["interim_file"])
    out_dir = out_dir or resolve_path(params["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_table(pd.read_parquet(in_file), params)
    parquet_file = out_dir / params["features"]["processed_file"]
    features.to_parquet(parquet_file)
    log_dataset_run(features, parquet_file, params, run_name="dataset-features")
    logger.info("Features: %d filas -> %s", len(features), parquet_file)
    return parquet_file


@click.command()
@click.option("--params", "params_path", default=None)
@click.option("--in-file", default=None, help="Sobrescribe interim parquet.")
@click.option("--out-dir", default=None, help="Sobrescribe paths.processed.")
def main(params_path, in_file, out_dir):
    """Parquet limpio (data/interim) -> tabla de features (data/processed)."""
    params = load_params(params_path)
    build_features_dataset(
        params,
        in_file=resolve_path(in_file) if in_file else None,
        out_dir=resolve_path(out_dir) if out_dir else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    load_dotenv(find_dotenv())
    main()
