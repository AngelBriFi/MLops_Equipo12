"""Datos sintéticos con la misma forma que los CSV de MiBici (solo para pruebas)."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_params


@pytest.fixture()
def params(tmp_path):
    p = load_params()
    root = tmp_path
    for key in ("raw", "interim", "processed", "models", "figures", "reports"):
        p["paths"][key] = str(root / key)
    p["mlflow"]["tracking_uri"] = f"sqlite:///{root/'mlflow.db'}"
    p["training"].update({"cv_splits": 2, "n_iter": 2, "search_sample_frac": 0.5})
    p["training"]["search_space"]["n_estimators"] = {"type": "int", "low": 5, "high": 10}
    return p


@pytest.fixture()
def raw_csvs(params):
    """Tres 'meses' de viajes de 4 estaciones con patrón horario + corrupción de columna."""
    rng = np.random.default_rng(0)
    raw = __import__("pathlib").Path(params["paths"]["raw"]) / "2026"
    raw.mkdir(parents=True)
    for m in (1, 2, 3):
        hours = pd.date_range(f"2026-{m:02d}-01", periods=24 * 28, freq="h")
        rows = []
        for st in (10, 11, 12, 13):
            lam = 1 + 3 * np.isin(hours.hour, (8, 18)) * (st - 9) / 2
            for h, n in zip(hours, rng.poisson(lam)):
                for _ in range(n):
                    s = h + pd.Timedelta(seconds=int(rng.integers(0, 3599)))
                    rows.append({"Viaje_Id": len(rows), "Usuario_Id": 1, "Genero": "M",
                                 "Inicio_del_viaje": s.strftime("%Y-%m-%d %H:%M:%S"),
                                 "Fin_del_viaje": (s + pd.Timedelta(minutes=9)).strftime(
                                     "%Y-%m-%d %H:%M:%S"),
                                 "Origen_Id": st, "Destino_Id": int(rng.integers(10, 14))})
        df = pd.DataFrame(rows)
        col = "Año_de_nacimiento" if m != 2 else "AÃ±o_de_nacimiento"   # nombres inconsistentes
        df[col] = 1990.0
        df.to_csv(raw / f"datos_abiertos_2026_{m:02d}.csv", index=False, encoding="latin1")
    return raw
