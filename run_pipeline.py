import subprocess

# run_pipeline
steps = [
    ["python", "src/data/make_dataset.py"],
    ["python", "src/features/build_features.py"],
    # ["python", "src/models/train.py"],
    # ["python", "src/models/evaluate.py"]
]

for step in steps:
    print(f"Running {' '.join(step)}")
    subprocess.run(step, check=True)

print("Pipeline completed")