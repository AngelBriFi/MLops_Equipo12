
"""Pruebas automatizadas de Data Drift para MiBici."""

import pandas as pd
import pytest

from src.config import load_params
from src.monitoring.data_drift import (
    simular_drift_demanda,
    detectar_drift,
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
    """Comprueba que la demanda histórica aumente un 30 %."""
    datos, params, columnas = datos_prueba

    alterados = simular_drift_demanda(datos, params)

    pd.testing.assert_frame_equal(
        alterados[columnas],
        datos[columnas] * 1.30,
    )


def test_datos_originales_no_cambian(datos_prueba):
    """Comprueba que la simulación no modifique los originales."""
    datos, params, _ = datos_prueba
    originales = datos.copy(deep=True)

    simular_drift_demanda(datos, params)

    pd.testing.assert_frame_equal(datos, originales)


def test_deteccion_drift(datos_prueba):
    """Comprueba que KS detecte un cambio de distribución."""
    datos, params, columnas = datos_prueba

    alterados = simular_drift_demanda(datos, params)

    resultados = detectar_drift(
        datos,
        alterados,
        columnas,
    )

    assert resultados["drift_detectado"].all()
    assert (resultados["p_valor"] < 0.05).all()


def test_sin_drift(datos_prueba):
    """Comprueba que no se detecte drift en datos idénticos."""
    datos, _, columnas = datos_prueba

    resultados = detectar_drift(
        datos,
        datos.copy(),
        columnas,
    )

    assert not resultados["drift_detectado"].any()


def test_incremento_invalido(datos_prueba):
    """Comprueba que se rechacen incrementos inferiores a -100 %."""
    datos, params, _ = datos_prueba

    with pytest.raises(ValueError):
        simular_drift_demanda(
            datos,
            params,
            incremento=-1.5,
        )
