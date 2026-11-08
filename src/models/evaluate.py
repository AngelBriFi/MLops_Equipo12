"""Métricas, líneas base y desgloses de error."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE y R2. Las predicciones se recortan a >= 0 (conteos)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def improvement_vs_baseline(metric: float, baseline_metric: float) -> float:
    """Mejora porcentual respecto al baseline (positivo = mejor)."""
    return float(100.0 * (baseline_metric - metric) / baseline_metric)


def station_volume_tiers(train: pd.DataFrame, id_col: str, target: str,
                         q: int = 3) -> pd.Series:
    """Clasifica estaciones por demanda media en train (proxy de 'zona')."""
    means = train.groupby(id_col)[target].mean()
    labels = ["baja", "media", "alta"] if q == 3 else None
    return pd.qcut(means.rank(method="first"), q=q, labels=labels)


def error_breakdown(test: pd.DataFrame, y_pred, by, target: str) -> pd.DataFrame:
    """MAE/RMSE por grupo (mes, hora, nivel de volumen...)."""
    df = pd.DataFrame({
        "group": by.values if hasattr(by, "values") else by,
        "y": test[target].values,
        "p": np.clip(np.asarray(y_pred, dtype=float), 0, None),
    })
    rows = []
    for g, part in df.groupby("group", observed=True):
        rows.append({"group": g, "n": len(part),
                     "mae": mean_absolute_error(part["y"], part["p"]),
                     "rmse": float(np.sqrt(mean_squared_error(part["y"], part["p"])))})
    return pd.DataFrame(rows)
