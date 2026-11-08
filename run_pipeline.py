"""Ejecuta el flujo completo: datos -> features -> entrenamiento.

Equivale a `make pipeline` y es el entry point `main` de MLproject
(`mlflow run .`). Cada etapa se invoca como módulo para que `src` sea importable.
"""
import subprocess
import sys

steps = [
    [sys.executable, "-m", "src.data.make_dataset"],
    [sys.executable, "-m", "src.features.build_features"],
    [sys.executable, "-m", "src.models.train_model"],
]

for step in steps:
    print(f"Running {' '.join(step)}")
    subprocess.run(step, check=True)

print("Pipeline completed")
