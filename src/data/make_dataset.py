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
import yaml

# Create hash for experiments ID
def file_hash(filepath):
    sha = hashlib.sha256()

    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)

    return sha.hexdigest()

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    

    logger = logging.getLogger('MAKE DATASET')
    logger.info('Making interim data set from raw data')

    #get to root project path
    project_dir = Path(__file__).resolve().parents[2]

    # Load config from folder data
    with open(f'{project_dir}/src/data/config.yaml', "r") as f:
        config = yaml.safe_load(f)

    fileName = config['fileNames']['interim']
    fileType = config['fileTypes']['interim']
    pathRaw = config['paths']['raw_data']
    pathInterim = config['paths']['interim']

    rawP = project_dir.joinpath(pathRaw)

    dirs = []
    for item in os.listdir(rawP):
        lclPath = Path(rawP).joinpath(item)
        for f in lclPath.glob('*.csv'):
            dirs.append(str(f))

    dfs = [pd.read_csv(path, encoding='latin1') for path in dirs if str(path).__contains__('datos_abiertos')]
    df = pd.concat(dfs, ignore_index=True)

    df = df.drop(columns=['AÃ±o_de_nacimiento','Año_de_nacimiento','Viaje_Id','Genero','Usuario_Id'])
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    df['Destino_Id'] = df['Destino_Id'].astype('str')
    df['Origen_Id'] = df['Origen_Id'].astype('str')
    df['Inicio_del_viaje'] = pd.to_datetime(df['Inicio_del_viaje'],format="%Y-%m-%d %H:%M:%S")
    df['Fin_del_viaje'] = pd.to_datetime(df['Fin_del_viaje'],format="%Y-%m-%d %H:%M:%S")

    interP = project_dir.joinpath(pathInterim)

    if not Path(interP).exists():
        os.mkdir(interP)

    if str(fileType).lower() == 'parquet':
        OUTPUT_FILE = f"{interP}/{fileName}.parquet"
        df.to_parquet(OUTPUT_FILE)
    elif str(fileType).lower() == 'csv':
        OUTPUT_FILE = f"{interP}/{fileName}.csv"
        df.to_csv(OUTPUT_FILE)
    else:
        raise Exception("Incorrect data type output, check the config yaml in /data")
    
    # mlflow.set_experiment("datasetToInterim")
    # with mlflow.start_run(nested=True):
    SCHEMA_FILE = Path(f"{interP}/schema.json")
    METADATA_FILE = Path(f"{interP}/metadata.json")

    schema =  pd.io.json.build_table_schema(df)

    with open(SCHEMA_FILE, "w") as f:
        json.dump(schema, f, indent=2)

    dataset_hash = file_hash(OUTPUT_FILE)

    metadata = {
        "rows": len(df),
        "columns": len(df.columns),
        "dataset_hash": dataset_hash
    }

    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    # mlflow.log_param("dataset_hash", dataset_hash)
    mlflow.set_tag("component", "make dataset")

    mlflow.log_metric("rows", len(df))
    mlflow.log_metric("columns", len(df.columns))
    mlflow.log_metric(
        "null_count",
        int(df.isna().sum().sum())
    )

    mlflow.log_artifact(str(OUTPUT_FILE))
    mlflow.log_artifact(str(SCHEMA_FILE))
    mlflow.log_artifact(str(METADATA_FILE))

    print("DATASET CREATED", interP)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
