
import logging
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    project_dir = Path(__file__).resolve().parents[2]

    with open(project_dir / "src/features/config.yaml", encoding="utf-8") as f:
        features_config = yaml.safe_load(f)

    with open(project_dir / "src/models/config.yaml", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)
    data_path = (
        project_dir
        / features_config["paths"]["processed_data"]
        / "processed_sample.parquet"
    )

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {data_path}")

    print("Cargando dataset...", flush=True)

    df = pd.read_parquet(data_path)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col != "demanda":
            df[col] = df[col].astype(np.float32)

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"])

    train_fraction = model_config["TrainingParameters"]["trainPerc"] / 100

    fechas = df["fecha_hora"].drop_duplicates().sort_values()
    corte = fechas.iloc[int(len(fechas) * train_fraction)]

    print(f"Fecha de corte: {corte}", flush=True)

    train_mask = (df["fecha_hora"] < corte).to_numpy()
    test_mask = ~train_mask

    categorical = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numeric = [
        col for col in df.select_dtypes(include=["number"]).columns
        if col != "demanda"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )

    print("Preparando datos de entrenamiento...", flush=True)

    X_train = preprocessor.fit_transform(df.loc[train_mask])
    y_train = df.loc[train_mask, "demanda"].to_numpy(dtype=np.float32)

    print(f"Matriz de entrenamiento: {X_train.shape}", flush=True)

    # Liberar columnas que ya no son necesarias para predecir
    X_test = preprocessor.transform(df.loc[test_mask])
    y_test = df.loc[test_mask, "demanda"].to_numpy(dtype=np.float32)

    del df

    if not sparse.issparse(X_train):
        X_train = sparse.csr_matrix(X_train)

    if not sparse.issparse(X_test):
        X_test = sparse.csr_matrix(X_test)

    print("Entrenando regresion lineal...", flush=True)

    model = LinearRegression()
    model.fit(X_train, y_train)

    del X_train, y_train

    print("Evaluando modelo...", flush=True)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Modelo: Regresion lineal")
    print(f"Fecha de corte: {corte}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    print("Registrando resultados en MLflow...", flush=True)

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

        pipeline_model = {
            "preprocessor": preprocessor,
            "model": model,
        }

        mlflow.sklearn.log_model(
            sk_model=model,
            name="linear_model",
        )

    print("Entrenamiento finalizado correctamente.", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

