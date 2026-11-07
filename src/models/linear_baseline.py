
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    project_dir = Path(__file__).resolve().parents[2]

    # Reutilizar las configuraciones existentes
    with open(project_dir / "src/features/config.yaml", encoding="utf-8") as f:
        features_config = yaml.safe_load(f)

    with open(project_dir / "src/models/config.yaml", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    processed_dir = project_dir / features_config["paths"]["processed_data"]
    processed_name = features_config["fileNames"]["processed"]
    data_path = processed_dir / f"{processed_name}.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontro {data_path}. "
            "Primero debe generarse el dataset con el pipeline existente."
        )

    df = pd.read_parquet(data_path)
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])
    df = df.sort_values("fecha_hora").copy()

    # Mismo criterio de separacion temporal que XGBoost
    train_fraction = model_config["TrainingParameters"]["trainPerc"] / 100
    fechas = df["fecha_hora"].drop_duplicates().sort_values()
    corte = fechas.iloc[int(len(fechas) * train_fraction)]

    train = df[df["fecha_hora"] < corte].copy()
    test = df[df["fecha_hora"] >= corte].copy()

    X_train = train.drop(columns=["demanda"])
    y_train = train["demanda"]
    X_test = test.drop(columns=["demanda"])
    y_test = test["demanda"]

    # Mismo esquema de variables que el entrenamiento existente
    categorical = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ]
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression()),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    with mlflow.start_run(run_name="mibici_linear_baseline"):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("train_fraction", train_fraction)
        mlflow.log_param("cutoff_date", str(corte))
        mlflow.log_param("dataset_path", str(data_path))
        mlflow.log_metrics({
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        })
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

    print("Modelo: Regresion lineal")
    print(f"Fecha de corte: {corte}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
