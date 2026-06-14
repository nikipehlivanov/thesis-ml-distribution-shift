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

# Data loading

def load_split(path):
    """
    Load one split file.

    Expected input:
    - train split
    - ID test split
    - OOD test split

    The date column is converted to datetime and the data is sorted.
    """
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {path}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column in {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(f"Invalid dates found in {path}")

    if df.empty:
        raise ValueError(f"Split file is empty: {path}")

    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_asset_splits(asset_name):
    """
    Load train, ID test, and OOD test splits for one asset.

    Returns:
    - train_df
    - id_test_df
    - ood_test_df
    """
    if asset_name not in ASSET_SPLITS:
        raise ValueError(f"Unknown asset name: {asset_name}")

    split_paths = ASSET_SPLITS[asset_name]

    train_df = load_split(split_paths["train"])
    id_test_df = load_split(split_paths["id_test"])
    ood_test_df = load_split(split_paths["ood_test"])

    return train_df, id_test_df, ood_test_df


# Prediction saving

def save_predictions(asset_name, model_name, period_name, test_df, predictions):
    """
    Save predictions for one model and one test period.

    Saved columns:
    - date
    - y_true
    - y_pred
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if len(predictions) != len(test_df):
        raise ValueError(
            f"Prediction length does not match test data length for "
            f"{asset_name}, {model_name}, {period_name}."
        )

    predictions_df = pd.DataFrame({
        "date": test_df["date"],
        "y_true": test_df[TARGET_COLUMN],
        "y_pred": predictions,
    })

    output_path = PREDICTIONS_DIR / f"{asset_name}_{model_name}_{period_name}_predictions.csv"

    predictions_df.to_csv(output_path, index=False)

    return output_path


# Single model experiment

def run_model_experiment(asset_name, model_name, model_function, train_df, id_test_df, ood_test_df):
    """
    Run one model on one asset.

    Steps:
    1. Fit model on train_df and predict ID test period.
    2. Fit model on train_df and predict OOD test period.
    3. Evaluate ID predictions.
    4. Evaluate OOD predictions.
    5. Compare ID vs OOD degradation.
    6. Save predictions.
    7. Return one metrics dictionary.
    """
    print(f"Running {model_name} for {asset_name}...")

    id_predictions = model_function(train_df, id_test_df)
    ood_predictions = model_function(train_df, ood_test_df)

    y_id_true = id_test_df[TARGET_COLUMN].values
    y_ood_true = ood_test_df[TARGET_COLUMN].values

    id_metrics = evaluate_predictions(y_id_true, id_predictions)
    ood_metrics = evaluate_predictions(y_ood_true, ood_predictions)

    comparison = compare_id_ood(id_metrics, ood_metrics)

    save_predictions(
        asset_name=asset_name,
        model_name=model_name,
        period_name="id_test",
        test_df=id_test_df,
        predictions=id_predictions,
    )

    save_predictions(
        asset_name=asset_name,
        model_name=model_name,
        period_name="ood_test",
        test_df=ood_test_df,
        predictions=ood_predictions,
    )

    result = {
        "asset": asset_name,
        "model": model_name,
        **comparison,
    }

    return result


# Asset-level experiment

def run_asset_experiments(asset_name):
    """
    Run all models for one asset.

    Returns:
    - list of metric dictionaries
    """
    print(f"\nLoading splits for {asset_name}...")

    train_df, id_test_df, ood_test_df = load_asset_splits(asset_name)

    print(f"Train rows: {len(train_df)}")
    print(f"ID test rows: {len(id_test_df)}")
    print(f"OOD test rows: {len(ood_test_df)}")

    asset_results = []

    for model_name, model_function in MODELS.items():
        result = run_model_experiment(
            asset_name=asset_name,
            model_name=model_name,
            model_function=model_function,
            train_df=train_df,
            id_test_df=id_test_df,
            ood_test_df=ood_test_df,
        )

        asset_results.append(result)

    return asset_results


# Full experiment pipeline

def run_all_experiments():
    """
    Run all models for all assets.

    Outputs:
    - results/model_metrics.csv
    - prediction CSV files in results/predictions/
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for asset_name in ASSET_SPLITS.keys():
        asset_results = run_asset_experiments(asset_name)
        all_results.extend(asset_results)

    results_df = metrics_to_dataframe(all_results)

    results_df.to_csv(METRICS_OUTPUT_PATH, index=False)

    print("\nAll experiments completed successfully.")
    print(f"Saved metrics to: {METRICS_OUTPUT_PATH}")
    print(f"Saved predictions to: {PREDICTIONS_DIR}")

    return results_df


if __name__ == "__main__":
    run_all_experiments()