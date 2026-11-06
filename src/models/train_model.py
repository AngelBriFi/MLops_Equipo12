import pandas as pd
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
import os
from pathlib import Path
import yaml
import logging
import mlflow.xgboost

def main():
    #Load data processed
    logger = logging.getLogger('TRAINING MODEL')
    logger.info('Training model in process')

    #get to root project path
    project_dir = Path(__file__).resolve().parents[2]

    # Load config from folder features
    with open(f'{project_dir}/src/features/config.yaml', "r") as f:
        config = yaml.safe_load(f)

    # Interim configuration
    fileNameProc = config['fileNames']['processed']
    fileTypeProc = config['fileTypes']['processed']
    pathProc = project_dir.joinpath(config['paths']['processed_data'])

    # Load config from folder data
    with open(f'{project_dir}\\src\\models\\config.yaml', "r") as f:
        trainingConfg = yaml.safe_load(f)

    # Model training configuration
    trainParam = trainingConfg['TrainingParameters']

    #LOAD DATA
    df = pd.read_parquet(f'{pathProc}\\{fileNameProc}.parquet')

    #CONVERT DATA
    for col in df.select_dtypes(include=["int64"]):
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]):
        df[col] = pd.to_numeric(df[col], downcast="float")

    trainPerc = trainParam['trainPerc']
    assert trainPerc < 90 and trainPerc > 10, 'Train split incorrect value, make sure is between 0-100'

    TRAIN_SPLIT = trainPerc/100.0

    # Ordenar por fecha
    demanda_modelo = df.sort_values("fecha_hora").copy()

    # Fechas únicas ordenadas: así no separas una misma hora entre train y test
    fechas_unicas = demanda_modelo["fecha_hora"].drop_duplicates().sort_values()

    # Posición de corte
    indice_corte = int(len(fechas_unicas) * TRAIN_SPLIT)

    # Fecha a partir de la cual inicia test
    fecha_corte = fechas_unicas.iloc[indice_corte]

    # Split temporal
    train = demanda_modelo[demanda_modelo["fecha_hora"] < fecha_corte].copy()
    test = demanda_modelo[demanda_modelo["fecha_hora"] >= fecha_corte].copy()

    # CREACION DE DATA SPLITS TRAIN Y TEST
    X_train = train.drop(columns=["demanda"])
    y_train = train["demanda"]

    X_test = test.drop(columns=["demanda"])
    y_test = test["demanda"]

    categorical_features = X_train.select_dtypes('object').columns
    numeric_features = X_train.select_dtypes('number').columns

    preprocessor = ColumnTransformer(transformers=[
                    ("num",StandardScaler(), numeric_features),
                    ("cat",OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features)])


    pipeline = Pipeline([ ('Preprocessing', preprocessor),
                         ('Model',XGBRegressor())])

    tscv = TimeSeriesSplit(n_splits=trainParam['splits'])

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', XGBRegressor(objective='reg:squarederror', device="cuda"))
    ])

    param_grid = {
        'model__n_estimators': trainParam['n_estimators'],
        'model__max_depth': trainParam['max_depth'],
        'model__learning_rate': trainParam['learning_rate'],
        'model__subsample': trainParam['subsample'],
        'model__colsample_bytree': trainParam['colsample_bytree']
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=trainParam['n_iterations'],
        cv=tscv,
        scoring=trainParam['scoring'],
        verbose=1
    )

    mlflow.set_tag("Training", "Training process")
    mlflow.xgboost.autolog()

    # START TEST
    search.fit(X_train, y_train)


    # estimators = trainParam['n_estimators']
    # depth = trainParam['max_depth']
    # lrate = trainParam['learning_rate']
    # subsample = trainParam['subsample']
    # colSampByTree = trainParam['colsample_bytree']
    # iterations = trainParam['n_iterations']
    # scoring = trainParam['scoring']
    # splits = trainParam['splits']

if __name__ == '__main__':
    main()