"""Gráficas de resultados (se guardan en reports/figures y se suben a MLflow)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_model_comparison(results: pd.DataFrame, out_file: Path) -> Path:
    """Barras de MAE y RMSE por modelo (incluye baselines)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("mae", "rmse")):
        ax.bar(results["model"], results[metric], color="#00a19b")
        ax.set_title(metric.upper() + " en prueba (menor es mejor)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_breakdown(breakdowns: dict[str, pd.DataFrame], title: str,
                   out_file: Path) -> Path:
    """MAE por grupo (mes, nivel de volumen) para cada modelo."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, bd in breakdowns.items():
        ax.plot(bd["group"].astype(str), bd["mae"], marker="o", label=name)
    ax.set_title(title)
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_pred_vs_actual(test: pd.DataFrame, preds: dict, station: str,
                        id_col: str, time_col: str, target: str,
                        out_file: Path, hours: int = 24 * 7) -> Path:
    """Serie real vs. predicha de una estación (última semana de prueba)."""
    mask = (test[id_col] == station).values
    sub = test.loc[mask].tail(hours)
    idx = sub.index
    pos = test.index.get_indexer(idx)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub[time_col], sub[target], label="real", color="black")
    for name, p in preds.items():
        ax.plot(sub[time_col], p[pos], label=name, alpha=0.8)
    ax.set_title(f"Estación {station}: demanda real vs. pronóstico")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file
