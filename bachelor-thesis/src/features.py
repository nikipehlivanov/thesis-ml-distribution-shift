from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SP500_INPUT = PROCESSED_DIR / "sp500_merged.csv"
NASDAQ_INPUT = PROCESSED_DIR / "nasdaq100_merged.csv"

SP500_OUTPUT = PROCESSED_DIR / "sp500_features.csv"
NASDAQ_OUTPUT = PROCESSED_DIR / "nasdaq100_features.csv"


RETURN_LAGS = 10
VOLATILITY_WINDOWS = [5, 10]


def load_processed_data(path):
    """
    Load a processed merged dataset.
    Expected columns:
    date, close, volume, vix_close
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {path}")

    if "close" not in df.columns:
        raise ValueError(f"Missing close column in {path}")

    if "vix_close" not in df.columns:
        raise ValueError(f"Missing vix_close column in {path}")

    if "volume" not in df.columns:
        raise ValueError(f"Missing volume column in {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["close", "volume", "vix_close"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("date").reset_index(drop=True)

    return df


def create_log_returns(df):
    """
    Create daily log returns.

    log_return_t = log(close_t / close_{t-1})
    """
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def create_return_lags(df, max_lag=RETURN_LAGS):
    """
    Create lagged return features.

    ret_lag_1 uses yesterday's return.
    ret_lag_10 uses return from ten trading days ago.
    """
    df = df.copy()

    for lag in range(1, max_lag + 1):
        df[f"ret_lag_{lag}"] = df["log_return"].shift(lag)

    return df


def create_volatility_features(df, windows=VOLATILITY_WINDOWS):
    """
    Create rolling volatility features from log returns.

    These use current and past returns only.
    """
    df = df.copy()

    for window in windows:
        df[f"volatility_{window}"] = df["log_return"].rolling(window=window).std()

    return df


def create_vix_features(df):
    """
    Create lagged VIX feature.

    vix_lag_1 uses yesterday's VIX value to avoid look-ahead bias.
    """
    df = df.copy()
    df["vix_lag_1"] = df["vix_close"].shift(1)
    return df


def create_volume_features(df):
    """
    Create volume change feature.

    volume_change_t = percentage change in volume from t-1 to t.
    """
    df = df.copy()
    df["volume_change"] = df["volume"].pct_change()
    return df


def create_target(df):
    """
    Create prediction target.

    target_t = next-day log return = log_return_{t+1}
    """
    df = df.copy()
    df["target"] = df["log_return"].shift(-1)
    return df


def get_feature_columns():
    """
    Return the exact feature columns used for modeling.
    """
    feature_cols = []

    for lag in range(1, RETURN_LAGS + 1):
        feature_cols.append(f"ret_lag_{lag}")

    for window in VOLATILITY_WINDOWS:
        feature_cols.append(f"volatility_{window}")

    feature_cols.extend([
        "vix_lag_1",
        "volume_change",
    ])

    return feature_cols


def validate_features(df, asset_name):
    """
    Validate final model-ready dataset.
    """
    required_cols = ["date", "log_return", "target"] + get_feature_columns()

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"{asset_name}: missing columns: {missing_cols}")

    if df.empty:
        raise ValueError(f"{asset_name}: feature dataset is empty.")

    if not df["date"].is_monotonic_increasing:
        raise ValueError(f"{asset_name}: dates are not sorted oldest to newest.")

    if df[required_cols].isna().any().any():
        missing_summary = df[required_cols].isna().sum()
        raise ValueError(
            f"{asset_name}: missing values found in final features:\n{missing_summary}"
        )

    feature_cols = get_feature_columns()

    for col in feature_cols + ["target"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{asset_name}: column is not numeric: {col}")

    print(f"{asset_name} validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Start date: {df['date'].min().date()}")
    print(f"End date: {df['date'].max().date()}")


def create_features(df, asset_name):
    """
    Full feature engineering pipeline for one asset.

    Input:
    cleaned merged dataset with close, volume, and vix_close.

    Output:
    model-ready supervised learning dataset.
    """
    df = df.copy()

    df = df.sort_values("date").reset_index(drop=True)

    df = create_log_returns(df)
    df = create_return_lags(df)
    df = create_volatility_features(df)
    df = create_vix_features(df)
    df = create_volume_features(df)
    df = create_target(df)

    final_cols = [
        "date",
        "close",
        "volume",
        "vix_close",
        "log_return",
    ] + get_feature_columns() + [
        "target",
    ]

    df = df[final_cols]

    # Drops first rows caused by lags/rolling/pct_change and last row caused by target shift
    df = df.dropna().reset_index(drop=True)

    validate_features(df, asset_name)

    return df


def create_all_features():
    """
    Create feature datasets for both assets.

    Inputs:
    - data/processed/sp500_merged.csv
    - data/processed/nasdaq100_merged.csv

    Outputs:
    - data/processed/sp500_features.csv
    - data/processed/nasdaq100_features.csv
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets...")
    sp500 = load_processed_data(SP500_INPUT)
    nasdaq = load_processed_data(NASDAQ_INPUT)

    print("Creating S&P 500 features...")
    sp500_features = create_features(sp500, "S&P 500")

    print("Creating NASDAQ-100 features...")
    nasdaq_features = create_features(nasdaq, "NASDAQ-100")

    print("Checking date alignment between final feature files...")

    common_dates = set(sp500_features["date"]).intersection(set(nasdaq_features["date"]))

    sp500_features = (
        sp500_features[sp500_features["date"].isin(common_dates)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    nasdaq_features = (
        nasdaq_features[nasdaq_features["date"].isin(common_dates)]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(sp500_features) != len(nasdaq_features):
        raise ValueError("Final feature datasets do not have the same number of rows.")

    if not sp500_features["date"].equals(nasdaq_features["date"]):
        raise ValueError("Final feature datasets are not date-aligned.")

    print("Final feature files are aligned.")
    print(f"Final rows: {len(sp500_features)}")
    print(f"Final start date: {sp500_features['date'].min().date()}")
    print(f"Final end date: {sp500_features['date'].max().date()}")

    print("Saving feature files...")
    sp500_features.to_csv(SP500_OUTPUT, index=False)
    nasdaq_features.to_csv(NASDAQ_OUTPUT, index=False)

    print(f"Saved: {SP500_OUTPUT}")
    print(f"Saved: {NASDAQ_OUTPUT}")

    return sp500_features, nasdaq_features


if __name__ == "__main__":
    create_all_features()