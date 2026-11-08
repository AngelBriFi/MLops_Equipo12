# -*- coding: utf-8 -*-
"""Entrenamiento, ajuste, evaluación y registro de modelos (MLflow).

Flujo:
  1. Split temporal train/test (sin mezclar el futuro con el pasado).
  2. Baselines sin entrenamiento (repetir lag 24 h / 7 d / media 24 h).
  3. Regresión lineal (referencia) y XGBoost (RandomizedSearchCV con
     TimeSeriesSplit sobre una submuestra de train; reajuste con todo train).
  4. Métricas en prueba + desglose por mes y nivel de volumen de estación.
  5. Todo queda en MLflow (run padre + runs hijos) y los modelos en el Model
     Registry; el mejor por MAE de validación cruzada recibe el alias champion.

Roles: ML Engineer (pipeline/registro), Data Scientist (selección y ajuste).
"""
from __future__ import annotations

import gc
import json
import logging
from pathlib import Path

import click
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import (RandomizedSearchCV, TimeSeriesSplit,
                                     cross_val_score)

from src.config import load_params, resolve_path
from src.models.evaluate import (error_breakdown, improvement_vs_baseline,
                                 regression_metrics, station_volume_tiers)
from src.models.pipeline import (MODEL_NAMES, build_pipeline, feature_columns,
                                 search_distributions)
from src.utils.helpers import git_commit, setup_mlflow, tracked_run
from src.visualization.visualize import (plot_breakdown, plot_model_comparison,
                                         plot_pred_vs_actual)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- datos ---
def downcast(df: pd.DataFrame, keep: tuple[str, ...] = ()) -> pd.DataFrame:
    """float64 -> float32 e int64 -> int32 (reduce a la mitad la memoria de la tabla)."""
    for col in df.columns:
        if col in keep:
            continue
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")
    return df


def temporal_split(df: pd.DataFrame, time_col: str, train_fraction: float):
    """Corte temporal por horas únicas: una misma hora nunca queda en ambos lados.

    No ordena la tabla completa (evita una copia de millones de filas): el corte
    es por máscara y solo se ordena el conjunto de prueba.
    """
    horas = df[time_col].drop_duplicates().sort_values()
    cutoff = horas.iloc[int(len(horas) * train_fraction)]
    mask = (df[time_col] < cutoff).to_numpy()
    train = df[mask]
    test = df[~mask].sort_values(time_col).reset_index(drop=True)
    return train, test, cutoff


def search_subsample(train: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Submuestra (ordenada en el tiempo) para acelerar la búsqueda."""
    frac = params["training"]["search_sample_frac"]
    sub = train.sample(frac=frac, random_state=params["seed"]) if frac < 1 else train
    return sub.sort_values(params["features"]["time_column"]).reset_index(drop=True)


# --------------------------------------------------------------- modelos ---
def evaluate_baselines(test: pd.DataFrame, params: dict) -> dict[str, dict]:
    """Pronósticos 'ingenuos': no se entrenan, usan columnas ya calculadas."""
    target = params["features"]["target"]
    window = params["features"]["rolling_window_hours"]
    out = {}
    for name, col in {"baseline_lag_24h": "lag_24h",
                      "baseline_lag_7d": "lag_7d",
                      f"baseline_media_{window}h": f"media_{window}h"}.items():
        out[name] = regression_metrics(test[target], test[col])
    return out


def tune_xgboost(X, y, params: dict) -> RandomizedSearchCV:
    """RandomizedSearchCV con validación cruzada temporal (refit=False)."""
    tcfg = params["training"]
    search = RandomizedSearchCV(
        build_pipeline("xgboost", params),
        param_distributions=search_distributions(params),
        n_iter=tcfg["n_iter"],
        scoring=tcfg["scoring"],
        cv=TimeSeriesSplit(n_splits=tcfg["cv_splits"]),
        random_state=params["seed"],
        n_jobs=1,          # XGBoost ya paraleliza internamente
        refit=False,
        verbose=1,
    )
    return search.fit(X, y)


def log_search_candidates(search: RandomizedSearchCV) -> None:
    """Cada candidato de la búsqueda queda como run hijo (para comparar en MLflow)."""
    res = search.cv_results_
    for i, cand in enumerate(res["params"]):
        with mlflow.start_run(run_name=f"xgb-candidate-{i:02d}", nested=True):
            mlflow.log_params({k.replace("model__", ""): v for k, v in cand.items()})
            mlflow.log_metric("cv_mae", float(-res["mean_test_score"][i]))
            mlflow.log_metric("cv_mae_std", float(res["std_test_score"][i]))
            mlflow.log_metric("rank", int(res["rank_test_score"][i]))
            mlflow.set_tag("candidate", "true")


def fit_log_register(name: str, pipe, train, test, feats, params, cv_mae,
                     tiers, extra_params=None):
    """Ajusta con todo train, evalúa en prueba y registra el modelo."""
    target = params["features"]["target"]
    reg_name = f"{params['mlflow']['registered_model_prefix']}-{name}"
    with mlflow.start_run(run_name=name, nested=True) as run:
        pipe.fit(train[feats], train[target])
        pred = pipe.predict(test[feats])
        metrics = regression_metrics(test[target], pred)
        metrics["cv_mae"] = cv_mae

        mlflow.log_param("model", name)
        mlflow.log_param("seed", params["seed"])
        mlflow.log_params(extra_params or {})
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if k != "cv_mae"})
        mlflow.log_metric("cv_mae", cv_mae)

        info = mlflow.sklearn.log_model(
            pipe, name="model", registered_model_name=reg_name,
            serialization_format="cloudpickle",
            signature=infer_signature(test[feats].head(100), pred[:100]),
            input_example=test[feats].head(5))
        version = getattr(info, "registered_model_version", None)
        mlflow.set_tag("registered_model", reg_name)
    return {"pipe": pipe, "pred": pred, "metrics": metrics, "run_id": run.info.run_id,
            "registered_model": reg_name, "version": version}


# ------------------------------------------------------------ orquestador ---
def run_training(params: dict, models: tuple[str, ...] = MODEL_NAMES,
                 features_file: Path | None = None) -> dict:
    p_id, p_time, target = (params["features"][k] for k in
                            ("id_column", "time_column", "target"))
    features_file = features_file or (resolve_path(params["paths"]["processed"])
                                      / params["features"]["processed_file"])
    figures_dir, models_dir, reports_dir = (
        resolve_path(params["paths"][k]) for k in ("figures", "models", "reports"))
    for d in (figures_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = downcast(pd.read_parquet(features_file), keep=(target,))
    feats = feature_columns(params)
    train, test, cutoff = temporal_split(df, p_time, params["split"]["train_fraction"])
    del df
    gc.collect()
    sub = search_subsample(train, params)
    logger.info("Train %d | Test %d | corte %s | submuestra búsqueda %d",
                len(train), len(test), cutoff, len(sub))

    tiers = station_volume_tiers(train, p_id, target)
    test_tier = test[p_id].map(tiers)
    setup_mlflow(params)
    tcfg = params["training"]
    results, preds, fitted = [], {}, {}

    with tracked_run("model-comparison") as parent:
        mlflow.set_tags({"git_commit": git_commit(), "features_file": str(features_file)})
        mlflow.log_params({"seed": params["seed"],
                           "train_fraction": params["split"]["train_fraction"],
                           "cutoff": str(cutoff), "n_train": len(train), "n_test": len(test),
                           "cv_splits": tcfg["cv_splits"], "n_iter": tcfg["n_iter"],
                           "search_sample_frac": tcfg["search_sample_frac"]})
        mlflow.log_artifact(str(resolve_path("params.yaml")))

        # 1) baselines (sin entrenamiento)
        for name, m in evaluate_baselines(test, params).items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model", name)
                mlflow.log_metrics({f"test_{k}": v for k, v in m.items()})
            results.append({"model": name, **m})
        base_mae = min(r["mae"] for r in results)
        base_rmse = min(r["rmse"] for r in results)

        # 2) regresión lineal
        if "linear" in models:
            pipe = build_pipeline("linear", params)
            cv = -cross_val_score(pipe, sub[feats], sub[target],
                                  cv=TimeSeriesSplit(n_splits=tcfg["cv_splits"]),
                                  scoring=tcfg["scoring"]).mean()
            fitted["linear"] = fit_log_register("linear", pipe, train, test, feats,
                                                params, float(cv), tiers)
        # 3) XGBoost
        best_params = {}
        if "xgboost" in models:
            search = tune_xgboost(sub[feats], sub[target], params)
            log_search_candidates(search)
            best_params = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                           for k, v in search.best_params_.items()}
            pipe = build_pipeline("xgboost", params)
            pipe.set_params(**{f"model__{k}": v for k, v in best_params.items()})
            fitted["xgboost"] = fit_log_register(
                "xgboost", pipe, train, test, feats, params,
                float(-search.best_score_), tiers, extra_params=best_params)

        # 4) resultados, champion y artefactos
        breakdowns_mes, breakdowns_vol = {}, {}
        for name, f in fitted.items():
            preds[name] = f["pred"]
            m = f["metrics"]
            results.append({"model": name, "mae": m["mae"], "rmse": m["rmse"],
                            "r2": m["r2"], "cv_mae": m["cv_mae"]})
            breakdowns_mes[name] = error_breakdown(test, f["pred"], test["mes"], target)
            breakdowns_vol[name] = error_breakdown(test, f["pred"], test_tier, target)
        res_df = pd.DataFrame(results)
        res_df["mejora_mae_vs_baseline_pct"] = res_df["mae"].map(
            lambda v: improvement_vs_baseline(v, base_mae))
        res_df["mejora_rmse_vs_baseline_pct"] = res_df["rmse"].map(
            lambda v: improvement_vs_baseline(v, base_rmse))

        champion = min(fitted, key=lambda n: fitted[n]["metrics"]["cv_mae"])
        client = MlflowClient()
        if fitted[champion]["version"] is not None:
            client.set_registered_model_alias(
                fitted[champion]["registered_model"],
                params["mlflow"]["champion_alias"], str(fitted[champion]["version"]))
        joblib.dump(fitted[champion]["pipe"], models_dir / "champion.joblib")
        mlflow.set_tag("champion", champion)

        res_df.to_csv(reports_dir / "model_comparison.csv", index=False)
        summary = {"champion": champion, "cutoff": str(cutoff), "best_xgboost_params": best_params,
                   "results": res_df.to_dict(orient="records")}
        (reports_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))

        if fitted:
            figs = [plot_model_comparison(res_df, figures_dir / "model_comparison.png"),
                    plot_breakdown(breakdowns_mes, "MAE por mes (prueba)",
                                   figures_dir / "mae_por_mes.png"),
                    plot_breakdown(breakdowns_vol, "MAE por nivel de volumen",
                                   figures_dir / "mae_por_volumen.png")]
            top_station = train.groupby(p_id)[target].mean().idxmax()
            figs.append(plot_pred_vs_actual(test, preds, top_station, p_id, p_time, target,
                                            figures_dir / "pred_vs_real.png"))
            for fig in figs:
                mlflow.log_artifact(str(fig))
            for tag, bd in (("mes", breakdowns_mes), ("volumen", breakdowns_vol)):
                for name, d in bd.items():
                    d.to_csv(reports_dir / f"error_{tag}_{name}.csv", index=False)
        mlflow.log_artifact(str(reports_dir / "model_comparison.csv"))
        mlflow.log_artifact(str(reports_dir / "metrics.json"))
        summary["parent_run_id"] = parent.info.run_id
    logger.info("Champion: %s\n%s", champion, res_df.to_string(index=False))
    return summary


@click.command()
@click.option("--params", "params_path", default=None)
@click.option("--model", "model_opt", type=click.Choice(["all", *MODEL_NAMES]), default="all")
@click.option("--features-file", default=None)
def main(params_path, model_opt, features_file):
    """Entrena, ajusta, evalúa y registra los modelos."""
    params = load_params(params_path)
    models = MODEL_NAMES if model_opt == "all" else (model_opt,)
    run_training(params, models, resolve_path(features_file) if features_file else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    load_dotenv(find_dotenv())
    main()
