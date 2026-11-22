
"""Simulación, detección y evaluación de Data Drift para MiBici."""

import pandas as pd
from scipy.stats import ks_2samp

from src.models.evaluate import regression_metrics
from src.models.pipeline import feature_columns


def simular_drift_demanda(
    datos: pd.DataFrame,
    params: dict,
    incremento: float = 0.30,
) -> pd.DataFrame:
    """Simula un aumento de demanda en las variables históricas."""

    if incremento < -1:
        raise ValueError("El incremento no puede ser menor que -100 %.")

    datos_drift = datos.copy()
    configuracion = params["features"]

    # Identificar variables relacionadas con la demanda
    columnas = list(configuracion["lags_hours"])

    columnas.append(
        f"media_{configuracion['rolling_window_hours']}h"
    )

    columnas.extend(
        f"rolling_max_{ventana}h"
        for ventana in configuracion.get("rolling_max_hours", [])
    )

    # Aplicar incremento a las variables históricas
    for columna in columnas:
        if columna not in datos_drift.columns:
            raise ValueError(
                f"Falta la columna requerida: {columna}"
            )

        datos_drift[columna] = (
            datos_drift[columna] * (1 + incremento)
        )

    return datos_drift


def detectar_drift(
    datos_originales: pd.DataFrame,
    datos_alterados: pd.DataFrame,
    columnas: list[str],
    umbral: float = 0.05,
) -> pd.DataFrame:
    """Detecta cambios estadísticos entre dos distribuciones."""

    if not 0 < umbral < 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")

    resultados = []

    for columna in columnas:
        original = datos_originales[columna].dropna()
        alterado = datos_alterados[columna].dropna()

        if original.empty or alterado.empty:
            raise ValueError(
                f"No hay suficientes datos para evaluar: {columna}"
            )

        estadistico, p_valor = ks_2samp(
            original,
            alterado,
        )

        resultados.append({
            "variable": columna,
            "estadistico_ks": float(estadistico),
            "p_valor": float(p_valor),
            "drift_detectado": bool(p_valor < umbral),
        })

    return pd.DataFrame(resultados)


def evaluar_impacto_drift(
    modelo,
    datos_originales: pd.DataFrame,
    datos_alterados: pd.DataFrame,
    params: dict,
) -> pd.DataFrame:
    """Compara el desempeño del modelo antes y después del drift."""

    columnas = feature_columns(params)
    objetivo = params["features"]["target"]

    if len(datos_originales) != len(datos_alterados):
        raise ValueError(
            "Los conjuntos deben tener el mismo tamaño."
        )

    # Generar predicciones con el mismo modelo
    pred_originales = modelo.predict(
        datos_originales[columnas]
    )

    pred_alteradas = modelo.predict(
        datos_alterados[columnas]
    )

    # Calcular métricas utilizando la demanda observada
    metricas_originales = regression_metrics(
        datos_originales[objetivo],
        pred_originales,
    )

    metricas_alteradas = regression_metrics(
        datos_alterados[objetivo],
        pred_alteradas,
    )

    # Comparar resultados
    return pd.DataFrame([
        {
            "escenario": "Sin drift",
            **metricas_originales,
        },
        {
            "escenario": "Con drift",
            **metricas_alteradas,
        },
    ])

