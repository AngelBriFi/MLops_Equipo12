
"""Pruebas automatizadas de Data Drift para MiBici."""

import pandas as pd
import pytest

from src.config import load_params
from src.models.pipeline import feature_columns
from src.monitoring.data_drift import (
    simular_drift_demanda,
    detectar_drift,
    evaluar_impacto_drift,
)


@pytest.fixture
def datos_prueba():
    """Genera datos sintéticos reproducibles."""
    params = load_params()
    f = params["features"]

    columnas = list(f["lags_hours"])
    columnas.append(f"media_{f['rolling_window_hours']}h")
    columnas.extend(
        f"rolling_max_{w}h"
        for w in f.get("rolling_max_hours", [])
    )

    datos = pd.DataFrame({
        columna: [float(i) for i in range(1, 101)]
        for columna in columnas
    })

    return datos, params, columnas


def test_simulacion_incremento_30(datos_prueba):
    datos, params, columnas = datos_prueba
    alterados = simular_drift_demanda(datos, params)

    pd.testing.assert_frame_equal(
        alterados[columnas],
        datos[columnas] * 1.30,
    )


def test_datos_originales_no_cambian(datos_prueba):
    datos, params, _ = datos_prueba
    originales = datos.copy(deep=True)

    simular_drift_demanda(datos, params)

    pd.testing.assert_frame_equal(datos, originales)


def test_deteccion_drift(datos_prueba):
    datos, params, columnas = datos_prueba
    alterados = simular_drift_demanda(datos, params)

    resultados = detectar_drift(datos, alterados, columnas)

    assert resultados["drift_detectado"].all()
    assert (resultados["p_valor"] < 0.05).all()


def test_sin_drift(datos_prueba):
    datos, _, columnas = datos_prueba

    resultados = detectar_drift(
        datos, datos.copy(), columnas
    )

    assert not resultados["drift_detectado"].any()


def test_incremento_invalido(datos_prueba):
    datos, params, _ = datos_prueba

    with pytest.raises(ValueError):
        simular_drift_demanda(datos, params, incremento=-1.5)


def test_evaluacion_impacto_drift(datos_prueba):
    """Verifica las métricas antes y después del drift."""
    datos, params, _ = datos_prueba

    datos["demanda"] = datos["lag_1h"] + 2.0

    for columna in feature_columns(params):
        if columna not in datos.columns:
            datos[columna] = 0.0

    class ModeloSimulado:
        def predict(self, X):
            return X["lag_1h"].to_numpy()

    alterados = simular_drift_demanda(datos, params)

    resultados = evaluar_impacto_drift(
        ModeloSimulado(),
        datos,
        alterados,
        params,
    )

    assert len(resultados) == 2
    assert resultados["mae"].notna().all()
    assert resultados["rmse"].notna().all()
    assert resultados["r2"].notna().all()

    assert resultados.iloc[1]["mae"] > resultados.iloc[0]["mae"]

