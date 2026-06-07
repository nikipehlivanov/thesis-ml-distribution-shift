from pathlib import Path
import pandas as pd
from metrics import evaluate_predictions, compare_id_ood, metrics_to_dataframe
from models.ar_model import train_and_predict as run_ar
from models.arimax_model import train_and_predict as run_arimax
from models.ridge_model import train_and_predict as run_ridge
from models.random_forest_model import train_and_predict as run_random_forest
from models.mlp_model import train_and_predict as run_mlp

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_OUTPUT_PATH = RESULTS_DIR / "model_metrics.csv"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Split file paths

ASSET_SPLITS = {
    "sp500": {
        "train": SPLITS_DIR / "sp500_train.csv",
        "id_test": SPLITS_DIR / "sp500_id_test.csv",
        "ood_test": SPLITS_DIR / "sp500_ood_test.csv",
    },
    "nasdaq100": {
        "train": SPLITS_DIR / "nasdaq100_train.csv",
        "id_test": SPLITS_DIR / "nasdaq100_id_test.csv",
        "ood_test": SPLITS_DIR / "nasdaq100_ood_test.csv",
    },
}

# Model registry

MODELS = {
    "AR": run_ar,
    "ARIMAX": run_arimax,
    "Ridge": run_ridge,
    "RandomForest": run_random_forest,
    "MLP": run_mlp,
}


TARGET_COLUMN = "target"

if __name__ == "__main__":
    #run_all_experiments()
    pass