from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


PIPELINE_STEPS = [
    "preprocessing.py",
    "features.py",
    "splits.py",
    "experiments.py",
    "results_analysis.py",
    "regime_analysis.py",
]


def run_step(script_name):
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    print(f"\nRunning {script_name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed.")

    print(f"Finished {script_name}")


def run_pipeline():
    print("Starting full thesis pipeline...")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\nFull thesis pipeline completed successfully.")


if __name__ == "__main__":
    run_pipeline()