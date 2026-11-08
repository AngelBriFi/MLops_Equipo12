# Diccionario de datos

## Crudo (CSV de MiBici)
| Columna | Uso |
|---|---|
| Inicio_del_viaje, Fin_del_viaje | Se conservan (timestamp `%Y-%m-%d %H:%M:%S`) |
| Origen_Id, Destino_Id | Se conservan (se tratan como texto/categoría) |
| Viaje_Id, Usuario_Id, Genero, Año_de_nacimiento | Descartadas (la de nacimiento llegó corrupta en julio, con dos nombres distintos) |

## Features (`data/processed/mibici_features.parquet`)
| Columna | Descripción |
|---|---|
| Origen_Id, fecha_hora | Estación y hora (granularidad estación × hora) |
| demanda | **Objetivo**: viajes que inician en la estación en esa hora |
| lag_1h, lag_2h, lag_6h, lag_12h, lag_24h, lag_7d | Demanda hace 1, 2, 6, 12 y 24 horas y hace 7 días |
| media_24h | Media de las 24 h previas (sin incluir la hora actual) |
| rolling_max_6h, rolling_max_12h, rolling_max_24h | Máximo de la demanda en las 6/12/24 h previas (por estación, sin incluir la hora actual) |
| hora, dia_semana (lunes=0), mes, es_fin_semana | Calendario |
| hour_sin, hour_cos, dow_sin, dow_cos, mes_sin, mes_cos | Codificación cíclica de hora, día de la semana y mes |
