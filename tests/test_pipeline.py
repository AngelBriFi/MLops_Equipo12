import pandas as pd
import pytest

from src.data.make_dataset import build_interim_dataset
from src.data.validate_dataset import validate_dataset
from src.features.build_features import build_features_dataset
from src.models.evaluate import regression_metrics
from src.models.pipeline import feature_columns, model_columns
from src.models.predict_model import load_model, predict
from src.models.train_model import run_training, temporal_split


def test_dataset_keeps_only_required_columns(params, raw_csvs):
    interim = build_interim_dataset(params)
    df = pd.read_parquet(interim)
    assert list(df.columns) == params["data"]["keep_columns"]
    assert df["Origen_Id"].dtype != "int64"      # ids como texto
    assert not df.duplicated().any()


def test_features_have_no_future_leakage(params, raw_csvs):
    build_interim_dataset(params)
    feats = pd.read_parquet(build_features_dataset(params))
    assert not feats.isna().any().any()
    st = feats[feats["Origen_Id"] == "10"].sort_values("fecha_hora")
    assert (st["lag_1h"].iloc[1:].values == st["demanda"].iloc[:-1].values).all()


def test_temporal_split_does_not_mix_hours(params, raw_csvs):
    build_interim_dataset(params)
    feats = pd.read_parquet(build_features_dataset(params))
    train, test, cutoff = temporal_split(feats, "fecha_hora", 0.8)
    assert train["fecha_hora"].max() < cutoff <= test["fecha_hora"].min()


def test_metrics_clip_negative_predictions():
    m = regression_metrics([0, 1, 2], [-5, 1, 2])
    assert m["mae"] < 5 / 3


@pytest.mark.parametrize("extended", [False, True], ids=["base", "ampliadas"])
def test_full_training_registers_models_and_champion(params, raw_csvs, extended):
    if extended:
        params["features"].update({"rolling_max_hours": [6, 12], "cyclical_time": True,
                                   "lags_hours": {"lag_1h": 1, "lag_2h": 2, "lag_6h": 6,
                                                  "lag_24h": 24, "lag_7d": 168}})
    build_interim_dataset(params)
    build_features_dataset(params)
    summary = run_training(params)
    assert summary["champion"] in ("linear", "xgboost")
    names = {r["model"] for r in summary["results"]}
    assert {"baseline_lag_24h", "baseline_lag_7d", "linear", "xgboost"} <= names
    model = load_model(params)
    feats = pd.read_parquet(build_features_dataset(params)).head(50)
    out = predict(model, feats, params)
    assert (out["demanda_pronosticada"] >= 0).all()
    assert len(out) == 50


def test_validate_dataset_rejects_end_before_start():
    bad = pd.DataFrame({
        "Inicio_del_viaje": ["2026-01-01 10:00:00"], "Fin_del_viaje": ["2026-01-01 09:00:00"],
        "Origen_Id": ["1"], "Destino_Id": ["2"]})
    with pytest.raises(ValueError):
        validate_dataset(bad)


def test_model_columns_follow_feature_config(params):
    params["features"].update({"rolling_max_hours": [], "cyclical_time": False})
    base = model_columns(params)
    assert "hora" in base["categorical"] and "rolling_max_6h" not in base["numeric"]
    params["features"].update({"rolling_max_hours": [6], "cyclical_time": True})
    ext = model_columns(params)
    assert "hora" not in ext["categorical"]
    assert {"rolling_max_6h", "hour_sin"} <= set(ext["numeric"])
    assert len(feature_columns(params)) == sum(len(v) for v in ext.values())


def test_rolling_max_stays_within_station_and_uses_only_the_past(params, raw_csvs):
    params["features"].update({"rolling_max_hours": [6], "cyclical_time": True})
    build_interim_dataset(params)
    feats = pd.read_parquet(build_features_dataset(params))
    assert {"rolling_max_6h", "hour_sin", "hour_cos", "mes_cos"} <= set(feats.columns)
    for st, part in feats.groupby("Origen_Id"):
        part = part.sort_values("fecha_hora").reset_index(drop=True)
        expected = part["demanda"].shift(1).rolling(6, min_periods=1).max()
        assert (part["rolling_max_6h"].iloc[10:] == expected.iloc[10:]).all(), st
