# Revisión del código del zip `MLops_Equipo12-main_hoy`

Notas para discutir en el equipo. Cada punto indica cómo lo comprobé. Ninguno invalida lo ya hecho; son cosas que conviene arreglar antes de entregar. En esta propuesta ya están corregidas, salvo donde se indica.

## Errores que afectan a las variables del modelo

| # | Dónde | Qué pasa | Cómo se comprobó | Qué hice |
|---|---|---|---|---|
| 1 | `build_features.py`, `Features()`: `origin_avg`, `origin_std`, `origin_max` | La asignación `df['origin_avg'] = df.groupby('Origen_Id')['demanda'].mean()` alinea por índice, pero la serie está indexada por estación y el DataFrame por número de fila. No coinciden, todo queda en `NaN` y `fillna(0)` lo convierte en 0. Las tres columnas son constantes y no aportan nada. Si se "arreglaran" tal cual, calcularían promedios con todo el periodo, incluido el de prueba, y filtrarían información del futuro. | Corrí su código sobre los 7,147,968 renglones reales: `nunique = 1` y `max = 0.0` en las tres columnas. | Las quité. Si quieren estadísticas por estación, deben calcularse solo con el entrenamiento. |
| 2 | `ValoresHistoricos()`: `lag_6h` | Usa `shift(3)`, o sea es un rezago de 3 horas con nombre de 6. | Lectura del código. | Ahora es `shift(6)`, definido en `params.yaml`. |
| 3 | `Features()`: `rolling_max_6h/12h/24h` | `groupby(...).shift(1).rolling(w).max()` aplica la ventana sobre toda la tabla, no por estación. En las primeras filas de cada estación la ventana incluye datos de la estación anterior. El efecto es pequeño (solo esas primeras filas). | Lectura del código. | Se calculan dentro de cada estación y hay una prueba unitaria que lo verifica. |

## Cosas que no corren o no se usan

| # | Dónde | Qué pasa |
|---|---|---|
| 4 | `MLproject`, entry `dataset` | Apunta a `make_dataset.py` en la raíz, que no existe. Corregido a `python -m src.data.make_dataset`. |
| 5 | `.github/workflows/mlflow-training.yml` | Ejecuta `python train.py`, que no existe. Falla en cada push a `main`. |
| 6 | `.github/workflows/mlflow-datasets.yml` | Corre en cada push sin datos en el runner, así que falla. Ambos workflows pasan a ejecución manual y se agrega un `ci.yml` con lint y pruebas. |
| 7 | `.gitignore` | Conserva marcadores de conflicto de merge (`<<<<<<< HEAD`, 14 líneas). Reescrito limpio. |
| 8 | `src/data/validate_dataset.py` | No lo llama ningún módulo. Ahora se ejecuta en `make_dataset`. Con los datos reales pasa sin errores (0 viajes con fin anterior al inicio). |
| 9 | `make_dataset.py` / `build_features.py` | Registran métricas sin un run abierto explícito (MLflow crea uno solo) y `dataset_hash` quedó comentado. Ahora cada etapa abre su run y registra el hash. |

## Lo que sí funciona bien y se conservó

- La detección automática de GPU en `train_model.py` (ahora `device: auto` en `params.yaml`).
- `mlflow run .` con `MLproject` y `run_pipeline.py`. Hubo que corregir cómo comparten el run padre las tres etapas; se probó de punta a punta con datos sintéticos.
- Las variables nuevas de su versión: lags adicionales, máximos móviles y codificación cíclica de hora, día y mes.

## ¿Mejoran sus variables al modelo? Comparación con los mismos hiperparámetros

Mismos datos, mismo corte temporal (2026-05-27), XGBoost con los mismos hiperparámetros en ambos casos:

| Variables | MAE | RMSE | R² |
|---|---|---|---|
| Base de la primera versión | 0.7363 | 1.3562 | 0.6853 |
| Las de ustedes, tal cual (con los errores 1 y 2) | 0.7325 | 1.3564 | 0.6852 |

Es un empate práctico (0.5 % de diferencia en MAE, RMSE idéntico). Los hiperparámetros se eligieron con el conjunto base, así que la comparación no es una ablación completa. Por eso el conjunto ampliado queda como valor por defecto en esta propuesta (aprovecha su trabajo) y es reversible: `cyclical_time: false` y `rolling_max_hours: []` en `params.yaml`.

## Limitación que afecta a todos los modelos

La malla estación × hora rellena con 0 las horas anteriores a que la estación existiera: 104 de 472 estaciones abren más de 60 días después del inicio, y el 18.6 % de las filas son de ese tipo. Hace más fácil el problema de lo que es. Propuesta para la siguiente iteración: filtrar las filas anteriores a la primera demanda de cada estación.
