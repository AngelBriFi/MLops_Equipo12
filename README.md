MLops Equipo 12 · Pronóstico de demanda por estación de MiBici
==============================

Predicción de la demanda horaria de bicicletas por estación (AMG) con datos abiertos de MiBici.
Proyecto basado en [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

## Reproducir de principio a fin

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # instala también el paquete local `src`
# 1) Descargar los CSV mensuales de https://mibici.net/en/open-data/ a data/raw/<año>/datos_abiertos_AAAA_MM.csv
make pipeline                            # data -> features -> train
make predict                             # inferencia con el modelo champion
make mlflow-ui                           # http://127.0.0.1:5000  (experimentos y Model Registry)
make test                                # pruebas unitarias con datos sintéticos
```

Hay tres formas equivalentes de correr el flujo completo:

| Forma | Comando |
|---|---|
| Make | `make pipeline` |
| Script | `python run_pipeline.py` |
| MLflow Projects | `mlflow run . --env-manager local` (con `-e dataset`, `-e features` o `-e training` para una sola etapa) |

Bajo `mlflow run .` cada etapa queda como run hijo del run del proyecto.

Todo parámetro (rutas, semilla `seed: 42`, lags, split, espacio de búsqueda) está en
[`params.yaml`](params.yaml); ningún módulo de `src/` los define internamente. Para usar GPU basta `device: auto`
(detecta CUDA y, si no hay, usa CPU).

> Los `config.yaml` de cada carpeta de `src/` (de la primera versión del equipo) ya no los lee el código;
> se conservan solo porque `notebooks/03-Model_Tests.ipynb` depende de ellos. Ver "Propuesta de consolidación" en el PR.

## Flujo y módulos

| Etapa | Módulo | Entrada → salida |
|---|---|---|
| Ingesta y limpieza | `src/data/make_dataset.py` (+ `validate_dataset.py`) | `data/raw/*.csv` → `data/interim/mibici_data.parquet` |
| Variables | `src/features/build_features.py` | interim → `data/processed/mibici_features.parquet` |
| Pipeline Scikit-Learn | `src/models/pipeline.py` | `ColumnTransformer` (StandardScaler + One-Hot) → modelo |
| Entrenamiento / ajuste / registro | `src/models/train_model.py` | features → MLflow + `models/champion.joblib` + `reports/` |
| Métricas y desgloses | `src/models/evaluate.py` | MAE, RMSE, R², mejora vs. baseline, error por mes y nivel de volumen |
| Inferencia | `src/models/predict_model.py` | features → `reports/predictions.parquet` |
| Gráficas | `src/visualization/visualize.py` | `reports/figures/*.png` |
| Utilidades | `src/config.py`, `src/utils/helpers.py` | configuración, hash de datos, esquema, runs de MLflow |

Los notebooks (`notebooks/`) exploran e importan estos módulos (`04-Modeling.ipynb`).

## Variables del modelo

Configurables en `features` de `params.yaml` (las columnas del modelo se derivan de ahí):

- Rezagos de 1, 2, 6, 12 y 24 horas y de 7 días; media móvil de 24 h; máximos móviles de 6, 12 y 24 h, todos **por estación** y usando solo el pasado.
- Hora, día de la semana y mes en codificación cíclica (seno/coseno), fin de semana y estación (One-Hot).
- Con `cyclical_time: false` y `rolling_max_hours: []` se recupera el conjunto base de la primera versión.

## Decisiones de modelado

- **Validación temporal**: corte 80/20 por fecha (sin mezclar futuro y pasado) y `TimeSeriesSplit` en la búsqueda de hiperparámetros.
- **Baselines**: repetir la demanda de hace 24 h, de hace 7 días y la media de 24 h. El criterio de éxito de la Fase 1 es superarlos.
- **Modelos**: regresión lineal (referencia interpretable) y XGBoost (relaciones no lineales; admite `count:poisson`, adecuado para conteos).
- **Ajuste**: `RandomizedSearchCV` sobre una submuestra de entrenamiento (`search_sample_frac`) y reajuste final con todo el entrenamiento. Cada candidato queda como run hijo en MLflow.
- **Selección del champion**: por MAE de validación cruzada (no por el conjunto de prueba, que queda como estimación final).
- **Predicciones**: se recortan a ≥ 0 porque la demanda es un conteo.

## MLflow

- Backend: `sqlite:///mlflow.db` (necesario para el Model Registry; se puede cambiar con `MLFLOW_TRACKING_URI`).
- Experimento `mibici-demand-forecast`: run `model-comparison` con runs hijos para cada baseline, `linear`, `xgboost` y cada candidato `xgb-candidate-NN`; además `dataset-interim` y `dataset-features` (hash, esquema y nulos).
- Model Registry: `mibici-demand-linear` y `mibici-demand-xgboost` (versión, hiperparámetros, métricas y firma); el ganador recibe el alias `champion`.
- Se registran además el commit de git, `params.yaml` y las gráficas como artefactos.

## Roles y su interacción

| Rol | Actividades en esta fase | Entrega a… |
|---|---|---|
| Data Engineer | `make_dataset.py` y `validate_dataset.py`: ingesta, validación de esquema y consistencia, limpieza y versionado del dataset (hash, esquema) | Data Scientist (parquet `interim`) |
| Data Scientist | `build_features.py`, definición de baselines y métricas, selección de modelos y espacio de búsqueda | ML Engineer (tabla de features + criterios de evaluación) |
| ML Engineer | `pipeline.py`, `train_model.py`: pipeline Scikit-Learn, ajuste, registro en MLflow, alias `champion` | MLOps / Operación (modelo registrado) |
| MLOps / DevOps | `params.yaml`, `MLproject`, Makefile, CI (`ci.yml`), pruebas, reproducibilidad, `predict_model.py` | Equipo (flujo reproducible) |

> Integrantes: Angel Irwin Briseño Fierro · David Rodrigo Alvarado Domínguez · Oscar Fitzgerald Barajas Matxo
> (completar en el PDF qué integrante cubrió cada rol).

## Estructura

```
├── params.yaml            <- configuración central
├── MLproject / run_pipeline.py   <- `mlflow run .`
├── Makefile               <- data, features, train, predict, test, lint, mlflow-ui
├── data/{raw,interim,processed,external}   <- no se versionan (solo .gitkeep)
├── docs/  models/  references/  reports/figures/
├── notebooks/             <- 01-EDA, 02-Features, 03-Model_Tests, 04-Modeling
├── src/{data,features,models,visualization,utils}  config.py
├── tests/                 <- pruebas con datos sintéticos
└── .github/workflows/     <- ci.yml (lint + pytest); datasets/training en ejecución manual
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
