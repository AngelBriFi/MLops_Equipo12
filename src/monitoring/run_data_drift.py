"""Ejecuta la evaluación de Data Drift utilizando el Champion XGBoost."""

from pathlib import Path

import joblib
import pandas as pd
import yaml

from src.models.evaluate import regression_metrics
from src.monitoring.data_drift import (
    detectar_drift,
    simular_drift_demanda,
)


RUTA_CHAMPION = Path(
    "mlruns/1/models/"
    "m-40935abc467b4d39a75d7234b0ad1ed7/"
    "artifacts/model.pkl"
)

RUTA_DATOS = Path("data/processed/processed_sample.parquet")
RUTA_CONFIGURACION = Path("params.yaml")
DIRECTORIO_RESULTADOS = Path("reports/data_drift")

TAMANO_MUESTRA = 5000
INCREMENTO_DRIFT = 0.30

COLUMNAS_DRIFT = [
    "lag_1h",
    "lag_2h",
    "lag_6h",
    "lag_24h",
    "lag_7d",
    "media_24h",
    "rolling_max_6h",
    "rolling_max_12h",
]


def main():
    print("Cargando Champion XGBoost...")

    champion = joblib.load(RUTA_CHAMPION)

    with RUTA_CONFIGURACION.open(encoding="utf-8") as archivo:
        params = yaml.safe_load(archivo)

    datos = pd.read_parquet(RUTA_DATOS)
    columnas_modelo = list(champion.feature_names_in_)

    muestra = (
        datos.sort_values("fecha_hora")
        .tail(TAMANO_MUESTRA)
        .copy()
    )

    columnas_requeridas = set(
        columnas_modelo + COLUMNAS_DRIFT + ["demanda", "fecha_hora"]
    )
    faltantes = columnas_requeridas.difference(muestra.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas necesarias: {sorted(faltantes)}"
        )

    print(f"Registros evaluados: {len(muestra)}")
    print(f"Incremento simulado: {INCREMENTO_DRIFT:.0%}")

    datos_drift = simular_drift_demanda(
        muestra,
        params,
        incremento=INCREMENTO_DRIFT,
    )

    print("\nEvaluando Champion...")

    predicciones_originales = champion.predict(
        muestra[columnas_modelo]
    )
    predicciones_drift = champion.predict(
        datos_drift[columnas_modelo]
    )

    metricas_originales = regression_metrics(
        muestra["demanda"],
        predicciones_originales,
    )
    metricas_drift = regression_metrics(
        muestra["demanda"],
        predicciones_drift,
    )

    resultados_metricas = pd.DataFrame([
        {
            "escenario": "Sin drift",
            **metricas_originales,
        },
        {
            "escenario": "Con drift",
            **metricas_drift,
        },
    ])

    print("\nDetectando Data Drift...")

    resultados_deteccion = detectar_drift(
        muestra,
        datos_drift,
        COLUMNAS_DRIFT,
    )

    DIRECTORIO_RESULTADOS.mkdir(
        parents=True,
        exist_ok=True,
    )

    ruta_metricas = (
        DIRECTORIO_RESULTADOS / "metricas_champion.csv"
    )
    ruta_deteccion = (
        DIRECTORIO_RESULTADOS / "deteccion_drift.csv"
    )

    resultados_metricas.to_csv(
        ruta_metricas,
        index=False,
    )
    resultados_deteccion.to_csv(
        ruta_deteccion,
        index=False,
    )

    print("\nMÉTRICAS DEL CHAMPION")
    print(resultados_metricas.to_string(index=False))

    print("\nDETECCIÓN DE DATA DRIFT")
    print(resultados_deteccion.to_string(index=False))

    print("\nRESULTADOS GUARDADOS")
    print(ruta_metricas)
    print(ruta_deteccion)

    print(
        "\nNota: esta prueba mide sensibilidad ante drift "
        "simulado. No se ha verificado que la muestra "
        "sea independiente del entrenamiento."
    )


if __name__ == "__main__":
    main()
