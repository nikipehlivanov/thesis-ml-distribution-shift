from pathlib import Path
import pandas as pd

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Input feature files

SP500_FEATURES_PATH = PROCESSED_DIR / "sp500_features.csv"
NASDAQ_FEATURES_PATH = PROCESSED_DIR / "nasdaq100_features.csv"

# Output split files

SP500_TRAIN_PATH = SPLITS_DIR / "sp500_train.csv"
SP500_ID_TEST_PATH = SPLITS_DIR / "sp500_id_test.csv"
SP500_OOD_TEST_PATH = SPLITS_DIR / "sp500_ood_test.csv"

NASDAQ_TRAIN_PATH = SPLITS_DIR / "nasdaq100_train.csv"
NASDAQ_ID_TEST_PATH = SPLITS_DIR / "nasdaq100_id_test.csv"
NASDAQ_OOD_TEST_PATH = SPLITS_DIR / "nasdaq100_ood_test.csv"

# Experimental periods

TRAIN_START = "2010-01-01"
TRAIN_END = "2017-12-31"

ID_TEST_START = "2018-01-01"
ID_TEST_END = "2019-12-31"

OOD_TEST_START = "2020-01-01"
OOD_TEST_END = "2021-12-31"

# Feature and target columns

FEATURE_COLUMNS = [
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
    "ret_lag_6",
    "ret_lag_7",
    "ret_lag_8",
    "ret_lag_9",
    "ret_lag_10",
    "volatility_5",
    "volatility_10",
    "vix_lag_1",
    "volume_change",
]

TARGET_COLUMN = "target"

def load_feature_data(path):
    """
    Load a feature dataset and prepare the date column.

    Input:
    - sp500_features.csv or nasdaq100_features.csv

    Output:
    - sorted DataFrame with datetime date column
    """
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {path}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column in {path}")

    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in {path}: {missing_features}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(f"Invalid dates found in {path}")

    df = df.sort_values("date").reset_index(drop=True)

    return df

def split_by_period(df):
    """
    Split one feature dataset into:
    - training period: 2010-2017
    - in-distribution stable test period: 2018-2019
    - out-of-distribution crisis test period: 2020-2021
    """
    df = df.copy()

    train_start = pd.to_datetime(TRAIN_START)
    train_end = pd.to_datetime(TRAIN_END)

    id_start = pd.to_datetime(ID_TEST_START)
    id_end = pd.to_datetime(ID_TEST_END)

    ood_start = pd.to_datetime(OOD_TEST_START)
    ood_end = pd.to_datetime(OOD_TEST_END)

    train_df = df[
        (df["date"] >= train_start) &
        (df["date"] <= train_end)
    ].copy()

    id_test_df = df[
        (df["date"] >= id_start) &
        (df["date"] <= id_end)
    ].copy()

    ood_test_df = df[
        (df["date"] >= ood_start) &
        (df["date"] <= ood_end)
    ].copy()

    train_df = train_df.sort_values("date").reset_index(drop=True)
    id_test_df = id_test_df.sort_values("date").reset_index(drop=True)
    ood_test_df = ood_test_df.sort_values("date").reset_index(drop=True)

    return train_df, id_test_df, ood_test_df

def validate_single_split(split_df, split_name, asset_name):
    """
    Validate one split.
    """
    if split_df.empty:
        raise ValueError(f"{asset_name}: {split_name} split is empty.")

    if not split_df["date"].is_monotonic_increasing:
        raise ValueError(f"{asset_name}: {split_name} dates are not sorted.")

    required_cols = ["date", TARGET_COLUMN] + FEATURE_COLUMNS

    missing_cols = [col for col in required_cols if col not in split_df.columns]
    if missing_cols:
        raise ValueError(
            f"{asset_name}: {split_name} missing columns: {missing_cols}"
        )

    if split_df[required_cols].isna().any().any():
        missing_summary = split_df[required_cols].isna().sum()
        raise ValueError(
            f"{asset_name}: {split_name} has missing values:\n{missing_summary}"
        )

def validate_splits(train_df, id_test_df, ood_test_df, asset_name):
    """
    Validate that splits are non-empty, sorted, and non-overlapping.
    """
    validate_single_split(train_df, "train", asset_name)
    validate_single_split(id_test_df, "ID test", asset_name)
    validate_single_split(ood_test_df, "OOD test", asset_name)

    max_train_date = train_df["date"].max()
    min_id_date = id_test_df["date"].min()

    max_id_date = id_test_df["date"].max()
    min_ood_date = ood_test_df["date"].min()

    if max_train_date >= min_id_date:
        raise ValueError(
            f"{asset_name}: train and ID test periods overlap."
        )

    if max_id_date >= min_ood_date:
        raise ValueError(
            f"{asset_name}: ID test and OOD test periods overlap."
        )

    train_dates = set(train_df["date"])
    id_dates = set(id_test_df["date"])
    ood_dates = set(ood_test_df["date"])

    if train_dates.intersection(id_dates):
        raise ValueError(f"{asset_name}: train and ID test share dates.")

    if train_dates.intersection(ood_dates):
        raise ValueError(f"{asset_name}: train and OOD test share dates.")

    if id_dates.intersection(ood_dates):
        raise ValueError(f"{asset_name}: ID test and OOD test share dates.")

    print(f"\n{asset_name} split validation passed.")
    print(
        f"Train:    {train_df['date'].min().date()} to "
        f"{train_df['date'].max().date()} | rows: {len(train_df)}"
    )
    print(
        f"ID test:  {id_test_df['date'].min().date()} to "
        f"{id_test_df['date'].max().date()} | rows: {len(id_test_df)}"
    )
    print(
        f"OOD test: {ood_test_df['date'].min().date()} to "
        f"{ood_test_df['date'].max().date()} | rows: {len(ood_test_df)}"
    )

def validate_cross_asset_alignment(sp500_splits, nasdaq_splits):
    """
    Validate that S&P 500 and NASDAQ-100 splits have matching dates.

    This is expected because the preprocessing stage aligned the datasets.
    """
    split_names = ["train", "ID test", "OOD test"]

    for split_name, sp500_df, nasdaq_df in zip(
        split_names,
        sp500_splits,
        nasdaq_splits
    ):
        if len(sp500_df) != len(nasdaq_df):
            raise ValueError(
                f"Cross-asset mismatch in {split_name}: "
                f"S&P rows={len(sp500_df)}, NASDAQ rows={len(nasdaq_df)}"
            )

        if not sp500_df["date"].equals(nasdaq_df["date"]):
            raise ValueError(
                f"Cross-asset date mismatch in {split_name}."
            )

    print("\nCross-asset alignment validation passed.")
    print("S&P 500 and NASDAQ-100 splits contain the same dates.")

def save_splits(train_df, id_test_df, ood_test_df, train_path, id_path, ood_path):
    """
    Save split datasets to CSV files.
    """
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    id_test_df.to_csv(id_path, index=False)
    ood_test_df.to_csv(ood_path, index=False)

    print(f"Saved: {train_path}")
    print(f"Saved: {id_path}")
    print(f"Saved: {ood_path}")

def create_splits_for_asset(input_path, train_path, id_path, ood_path, asset_name):
    """
    Full split pipeline for one asset.
    """
    print(f"\nCreating splits for {asset_name}...")

    df = load_feature_data(input_path)

    train_df, id_test_df, ood_test_df = split_by_period(df)

    validate_splits(train_df, id_test_df, ood_test_df, asset_name)

    save_splits(
        train_df,
        id_test_df,
        ood_test_df,
        train_path,
        id_path,
        ood_path,
    )

    return train_df, id_test_df, ood_test_df

def create_all_splits():
    """
    Create train / ID test / OOD test splits for both assets.

    Inputs:
    - data/processed/sp500_features.csv
    - data/processed/nasdaq100_features.csv

    Outputs:
    - data/splits/sp500_train.csv
    - data/splits/sp500_id_test.csv
    - data/splits/sp500_ood_test.csv
    - data/splits/nasdaq100_train.csv
    - data/splits/nasdaq100_id_test.csv
    - data/splits/nasdaq100_ood_test.csv
    """
    sp500_splits = create_splits_for_asset(
        SP500_FEATURES_PATH,
        SP500_TRAIN_PATH,
        SP500_ID_TEST_PATH,
        SP500_OOD_TEST_PATH,
        "S&P 500",
    )

    nasdaq_splits = create_splits_for_asset(
        NASDAQ_FEATURES_PATH,
        NASDAQ_TRAIN_PATH,
        NASDAQ_ID_TEST_PATH,
        NASDAQ_OOD_TEST_PATH,
        "NASDAQ-100",
    )

    validate_cross_asset_alignment(sp500_splits, nasdaq_splits)

    print("\nAll train/test splits created successfully.")

    return sp500_splits, nasdaq_splits

if __name__ == "__main__":
    create_all_splits()