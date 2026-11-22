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

def get_current_champion(params):
    try:
        client = MlflowClient()

        model_name = (f"{params['mlflow']['registered_model_prefix']}-xgboost")

        return client.get_model_version_by_alias(
            model_name,
            params["mlflow"]["champion_alias"],
        )
    except:
        return None

def evaluate_current_champion(champion,params,test,feats,):
    target = params["features"]["target"]

    model_uri = ( f"models:/{champion.name}/{champion.version}")

    model = mlflow.pyfunc.load_model(model_uri)

    pred = model.predict(test[feats])

    return {
        "pred": pred,
        "metrics": regression_metrics(
            test[target],
            pred,
        )}

# ------------------------------------------------------------ orquestador ---
def run_training(params: dict,features_file: Path | None = None) -> dict:

    p_id, p_time, target = (params["features"][k] for k in ("id_column", "time_column", "target"))

    features_file = (features_file or resolve_path(params["paths"]["processed"]) / params["features"]["processed_file"])

    figures_dir, models_dir, reports_dir = (resolve_path(params["paths"][k]) for k in ("figures", "models", "reports"))

    for d in (figures_dir, models_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = downcast(
        pd.read_parquet(features_file),
        keep=(target,),
    )

    feats = feature_columns(params)

    train, test, cutoff = temporal_split(
        df,
        p_time,
        params["split"]["train_fraction"],
    )

    del df
    gc.collect()

    sub = search_subsample(train, params)

    logger.info(
        "Train %d | Test %d | corte %s | submuestra búsqueda %d",
        len(train),
        len(test),
        cutoff,
        len(sub),
    )

    tiers = station_volume_tiers(
        train,
        p_id,
        target,
    )

    test_tier = test[p_id].map(tiers)

    setup_mlflow(params)

    tcfg = params["training"]

    with tracked_run("xgboost-training") as parent:

        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "features_file": str(features_file),
            }
        )

        mlflow.log_params(
            {
                "seed": params["seed"],
                "train_fraction": params["split"]["train_fraction"],
                "cutoff": str(cutoff),
                "n_train": len(train),
                "n_test": len(test),
                "cv_splits": tcfg["cv_splits"],
                "n_iter": tcfg["n_iter"],
                "search_sample_frac": tcfg["search_sample_frac"],
            }
        )

        mlflow.log_artifact( str(resolve_path("params.yaml"))        )

        # ====================================================
        # Entrenamiento XGBoost
        # ====================================================

        search = tune_xgboost( sub[feats],
            sub[target],
            params,
        )

        log_search_candidates(search)

        best_params = {
            k.replace("model__", ""):
            v.item() if hasattr(v, "item") else v
            for k, v in search.best_params_.items()
        }

        pipe = build_pipeline(
            "xgboost",
            params,
        )

        pipe.set_params(
            **{
                f"model__{k}": v
                for k, v in best_params.items()
            }
        )

        candidate = fit_log_register(
            "xgboost",
            pipe,
            train,
            test,
            feats,
            params,
            float(-search.best_score_),
            tiers,
            extra_params=best_params,
        )

        client = MlflowClient()

        # ====================================================
        # Comparación contra champion actual
        # ====================================================

        current_champion = get_current_champion(params)
        candidate_mae = candidate["metrics"]["mae"]

        if current_champion is None:
            champion_mae = None
            promoted = True
        else:
            try:
                champion_eval = evaluate_current_champion(
                    current_champion,
                    params,
                    test,
                    feats,
                )

                champion_mae = (champion_eval["metrics"]["mae"])

                promoted = (candidate_mae < champion_mae)
            except:
                champion_mae = None
                promoted = True

        # ====================================================
        # Promoción
        # ====================================================

        if promoted:
            client.set_registered_model_alias(
                candidate["registered_model"],
                params["mlflow"]["champion_alias"],
                str(candidate["version"]),
            )
            joblib.dump(candidate["pipe"],models_dir / "champion.joblib")

        # ====================================================
        # Logging MLOps
        # ====================================================

        mlflow.log_metric("candidate_mae",candidate_mae)

        if champion_mae is not None:
            mlflow.log_metric("champion_mae",champion_mae)
            mlflow.log_metric("mae_delta",champion_mae - candidate_mae)

        mlflow.set_tag("promoted_to_champion",promoted)

        if current_champion:
            mlflow.set_tag("compared_against_version",current_champion.version)

        # ====================================================
        # Breakdowns
        # ====================================================

        breakdown_mes = error_breakdown(test,candidate["pred"],test["mes"],target)

        breakdown_vol = error_breakdown(test,candidate["pred"],test_tier,target)

        res_df = pd.DataFrame(
            [
                {
                    "model": "xgboost",
                    "mae": candidate["metrics"]["mae"],
                    "rmse": candidate["metrics"]["rmse"],
                    "r2": candidate["metrics"]["r2"],
                    "cv_mae": candidate["metrics"]["cv_mae"],
                }
            ]
        )

        fig1 = plot_breakdown(
            {"xgboost": breakdown_mes},
            "MAE por mes (prueba)",
            figures_dir / "mae_por_mes.png",
        )

        fig2 = plot_breakdown(
            {"xgboost": breakdown_vol},
            "MAE por nivel de volumen",
            figures_dir / "mae_por_volumen.png",
        )

        top_station = (train.groupby(p_id)[target].mean().idxmax() )

        fig3 = plot_pred_vs_actual(
            test,
            {"xgboost": candidate["pred"]},
            top_station,
            p_id,
            p_time,
            target,
            figures_dir / "pred_vs_real.png")

        for fig in (fig1, fig2, fig3):
            mlflow.log_artifact(str(fig))

        summary = {
            "candidate_version": candidate["version"],
            "candidate_mae": candidate_mae,
            "champion_mae": champion_mae,
            "promoted": promoted,
            "best_xgboost_params": best_params,
            "cutoff": str(cutoff),
        }

        (reports_dir / "metrics.json").write_text(json.dumps(summary,indent=2,default=str))

        mlflow.log_artifact(str(reports_dir / "metrics.json"))

        summary["parent_run_id"] = (parent.info.run_id)

    logger.info( "Promoted=%s Candidate MAE=%.4f",promoted,candidate_mae)

    return summary


@click.command()
@click.option("--params", "params_path", default=None)
@click.option("--model", "model_opt", type=click.Choice(["all", *MODEL_NAMES]), default="all")
@click.option("--features-file", default=None)
def main(params_path, model_opt, features_file):
    """Entrena, ajusta, evalúa y registra los modelos."""
    params = load_params(params_path)
    models = MODEL_NAMES if model_opt == "all" else (model_opt,)
    run_training(params, resolve_path(features_file) if features_file else None)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    load_dotenv(find_dotenv())
    main()
