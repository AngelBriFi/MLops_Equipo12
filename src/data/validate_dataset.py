import pandas as pd


REQUIRED_COLUMNS = [
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]


def validate_dataset(df: pd.DataFrame) -> None:
    """Valida la estructura y consistencia del dataset MiBici."""

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas obligatorias: {missing_columns}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Se encontraron valores nulos en columnas obligatorias."
        )

    start_dates = pd.to_datetime(
        df["Inicio_del_viaje"], errors="coerce"
    )

    end_dates = pd.to_datetime(
        df["Fin_del_viaje"], errors="coerce"
    )

    if start_dates.isna().any() or end_dates.isna().any():
        raise ValueError(
            "Se encontraron fechas inválidas."
        )

    if (end_dates < start_dates).any():
        raise ValueError(
            "Existen viajes cuya fecha de fin es anterior al inicio."
        )

    print("Validación completada: dataset consistente.")
