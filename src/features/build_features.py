# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
from pathlib import Path
import hashlib
import json
import mlflow

def file_hash(filepath):
    sha = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()

def CrearDemanda(df_init : pd.DataFrame):
    df = df_init.copy()
    demanda = (
    df.groupby([
        "Origen_Id",
        pd.Grouper(key="Inicio_del_viaje", freq="h")
    ])
    .size()
    .reset_index(name="demanda")
    )

    demanda = demanda.rename(columns={"Inicio_del_viaje": "fecha_hora"})
    return demanda.sort_values(["Origen_Id", "fecha_hora"])

def DemandasCasosCero(df_init : pd.DataFrame):
    df = df_init.copy()
    estaciones = df["Origen_Id"].unique()

    rango_fechas = pd.date_range(
        start=df["fecha_hora"].min().floor("h"),
        end=df["fecha_hora"].max().floor("h"),
        freq="h"
    )

    indice_completo = pd.MultiIndex.from_product(
        [estaciones, rango_fechas],
        names=["Origen_Id", "fecha_hora"]
    )

    demanda = (
        df.set_index(["Origen_Id", "fecha_hora"])
        .reindex(indice_completo, fill_value=0) # A los campos vacios entre filas se rellena con valor 0 en demanda
        .reset_index()
    )

    return demanda

def ValoresHistoricos(df_init : pd.DataFrame):
    df = df_init.copy()
    df = df.sort_values(["Origen_Id", "fecha_hora"])

    df["lag_1h"] = df.groupby("Origen_Id")["demanda"].shift(1)
    df["lag_24h"] = df.groupby("Origen_Id")["demanda"].shift(24)
    df["lag_7d"] = df.groupby("Origen_Id")["demanda"].shift(24 * 7)

    df["media_24h"] = (
        df.groupby("Origen_Id")["demanda"]
        .transform(lambda x: x.shift(1).rolling(24).mean())
    )
    return df.dropna().copy()

def Features(df_init : pd.DataFrame):
    df = df_init.copy()
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    df["hora"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.day_of_week
    df["mes"] = df["fecha_hora"].dt.month
    df["dia_mes"] = df["fecha_hora"].dt.day
    df["es_fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)
    return df

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath = 'data/interim/mibici_data.parquet', output_filepath = 'data/processed'):
    """ Runs data processing script to create the features from interim data
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from interim data')

    #get to root project path
    project_dir = Path(__file__).resolve().parents[2]

    df = pd.read_parquet(input_filepath)

    # PROCESAMIENTO
    # 1- Creacion de dataframe demanda
    demandaDF = CrearDemanda(df)
    del df

    # 2- Rellenar casos eliminados con demanda a valor 0
    demandaDF_2 = DemandasCasosCero(demandaDF)
    del demandaDF

    # 3- Crear los valores historicos por hora, dia y semana
    demandaDF_3 = ValoresHistoricos(demandaDF_2)
    del demandaDF_2

    # 4- Creacion de caracteristicas 
    demandaDF = Features(demandaDF_3)

    PARQUET_FILE = f"{output_filepath}/mibici_features.parquet"
    demandaDF.to_parquet(PARQUET_FILE)
    
    mlflow.set_experiment("dataset-versioning")
    with mlflow.start_run():
        SCHEMA_FILE = Path(f"{output_filepath}/schema.json")
        METADATA_FILE = Path(f"{output_filepath}/metadata.json")

        schema =  pd.io.json.build_table_schema(demandaDF)

        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema, f, indent=2)

        dataset_hash = file_hash(PARQUET_FILE)

        metadata = {
            "rows": len(demandaDF),
            "columns": len(demandaDF.columns),
            "dataset_hash": dataset_hash
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        mlflow.log_param("dataset_hash", dataset_hash)

        mlflow.log_metric("rows", len(demandaDF))
        mlflow.log_metric("columns", len(demandaDF.columns))
        mlflow.log_metric(
            "null_count",
            int(demandaDF.isna().sum().sum())
        )

        mlflow.log_artifact(str(PARQUET_FILE))
        mlflow.log_artifact(str(SCHEMA_FILE))
        mlflow.log_artifact(str(METADATA_FILE))

    print("FEATURES CREATED", output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
