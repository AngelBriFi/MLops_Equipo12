# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
import gzip
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

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #get to root project path
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    rawP = project_dir.joinpath(input_filepath)

    dirs = []
    for item in os.listdir(rawP):
        lclPath = Path(rawP).joinpath(item)
        for f in lclPath.glob('*.csv'):
            dirs.append(str(f))

    dfs = [pd.read_csv(path, encoding='latin1') for path in dirs if str(path).__contains__('datos_abiertos')]
    df = pd.concat(dfs, ignore_index=True)
    df.drop(columns=['AÃ±o_de_nacimiento','Año_de_nacimiento','Viaje_Id','Genero','Usuario_Id'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Destino_Id'] = df['Destino_Id'].astype('str')
    df['Origen_Id'] = df['Origen_Id'].astype('str')
    df['Inicio_del_viaje'] = pd.to_datetime(df['Inicio_del_viaje'],format="%Y-%m-%d %H:%M:%S")
    df['Fin_del_viaje'] = pd.to_datetime(df['Fin_del_viaje'],format="%Y-%m-%d %H:%M:%S")

    PARQUET_FILE = f"{output_filepath}/mibici_data.parquet"
    df.to_parquet(PARQUET_FILE)
    
    mlflow.set_experiment("dataset-versioning")
    with mlflow.start_run():
        SCHEMA_FILE = Path(f"{output_filepath}/schema.json")
        METADATA_FILE = Path(f"{output_filepath}/metadata.json")

        schema =  pd.io.json.build_table_schema(df)

        with open(SCHEMA_FILE, "w") as f:
            json.dump(schema, f, indent=2)

        dataset_hash = file_hash(PARQUET_FILE)

        metadata = {
            "rows": len(df),
            "columns": len(df.columns),
            "dataset_hash": dataset_hash
        }

        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # MLflow dataset tracking
        # dataset = mlflow.data.from_pandas(
        #     df,
        #     source=str(RAW_FILE),
        #     name="trips_dataset"
        # )

        # mlflow.log_input(dataset)

        mlflow.log_param("dataset_hash", dataset_hash)

        mlflow.log_metric("rows", len(df))
        mlflow.log_metric("columns", len(df.columns))
        mlflow.log_metric(
            "null_count",
            int(df.isna().sum().sum())
        )

        mlflow.log_artifact(str(PARQUET_FILE))
        mlflow.log_artifact(str(SCHEMA_FILE))
        mlflow.log_artifact(str(METADATA_FILE))

    print("DATASET CREATED", output_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
