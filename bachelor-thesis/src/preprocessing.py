from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

START_DATE = "2005-01-01"
END_DATE = "2024-12-31"

SP500_RAW = RAW_DIR / "sp500_futures.csv"
NASDAQ_RAW = RAW_DIR / "nasdaq100_futures.csv"
VIX_RAW = RAW_DIR / "vix.csv"

SP500_OUTPUT = PROCESSED_DIR / "sp500_merged.csv"
NASDAQ_OUTPUT = PROCESSED_DIR / "nasdaq100_merged.csv"


def parse_volume(value):
    """
    Convert volume strings such as:
    516.85K -> 516850
    1.2M -> 1200000
    5,000 -> 5000
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().replace(",", "")

    if value == "" or value == "-":
        return pd.NA

    multiplier = 1

    if value.endswith("K"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000
        value = value[:-1]

    try:
        return float(value) * multiplier
    except ValueError:
        return pd.NA


def clean_numeric_column(series):
    """
    Convert numeric columns stored as strings into floats.
    Handles commas and percent signs.
    """
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": pd.NA, "-": pd.NA})
        .astype(float)
    )


def load_raw_csv(path):
    """
    Load a raw CSV file.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def standardize_market_data(df, is_vix=False):
    """
    Standardize raw Investing.com-style CSV data.

    Expected possible columns:
    Date, Price, Open, High, Low, Vol., Change %

    Output:
    date, close, open, high, low, volume
    or for VIX:
    date, vix_close
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(".", "", regex=False)
        .str.replace("%", "pct", regex=False)
    )

    rename_map = {
        "date": "date",
        "price": "close",
        "close": "close",
        "open": "open",
        "high": "high",
        "low": "low",
        "vol": "volume",
        "change_pct": "change_pct",
    }

    df = df.rename(columns=rename_map)

    if "date" not in df.columns:
        raise ValueError("Missing date column.")

    if "close" not in df.columns:
        raise ValueError("Missing close/price column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = ["close", "open", "high", "low", "change_pct"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = clean_numeric_column(df[col])

    if "volume" in df.columns:
        df["volume"] = df["volume"].apply(parse_volume)

    df = df.dropna(subset=["date", "close"])

    # Important: reverse/sort raw data into oldest-to-newest order
    df = df.sort_values("date").reset_index(drop=True)

    df = df.drop_duplicates(subset=["date"], keep="last")

    if is_vix:
        df = df[["date", "close"]].rename(columns={"close": "vix_close"})
    else:
        keep_cols = ["date", "close"]

        for optional_col in ["open", "high", "low", "volume"]:
            if optional_col in df.columns:
                keep_cols.append(optional_col)

        df = df[keep_cols]

    return df


def filter_date_range(df, start_date=START_DATE, end_date=END_DATE):
    """
    Keep observations inside the chosen thesis sample period.
    """
    df = df.copy()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    df = df[(df["date"] >= start_date) & (df["date"] <= end_date)]

    return df.sort_values("date").reset_index(drop=True)


def merge_with_vix(asset_df, vix_df):
    """
    Merge one futures dataset with VIX using an inner join.

    Inner join is used to avoid artificial filling and to keep only dates
    available in both datasets.
    """
    merged = pd.merge(
        asset_df,
        vix_df,
        on="date",
        how="inner",
    )

    merged = merged.sort_values("date").reset_index(drop=True)

    return merged


def align_all_dates(sp500_df, nasdaq_df, vix_df):
    """
    Keep only dates that exist in all three datasets:
    S&P 500 futures, NASDAQ-100 futures, and VIX.

    This avoids calendar mismatches caused by weekends, holidays,
    or missing observations in one source.
    """
    common_dates = (set(sp500_df["date"]).intersection(set(nasdaq_df["date"])).intersection(set(vix_df["date"])))

    sp500_aligned = sp500_df[sp500_df["date"].isin(common_dates)].copy()
    nasdaq_aligned = nasdaq_df[nasdaq_df["date"].isin(common_dates)].copy()
    vix_aligned = vix_df[vix_df["date"].isin(common_dates)].copy()

    sp500_aligned = sp500_aligned.sort_values("date").reset_index(drop=True)
    nasdaq_aligned = nasdaq_aligned.sort_values("date").reset_index(drop=True)
    vix_aligned = vix_aligned.sort_values("date").reset_index(drop=True)

    return sp500_aligned, nasdaq_aligned, vix_aligned


def validate_processed_data(sp500_df, nasdaq_df):
    """
    Validate that final processed files are safe to use for feature engineering.
    """
    if len(sp500_df) == 0:
        raise ValueError("S&P 500 processed dataset is empty.")

    if len(nasdaq_df) == 0:
        raise ValueError("NASDAQ-100 processed dataset is empty.")

    if len(sp500_df) != len(nasdaq_df):
        raise ValueError("S&P 500 and NASDAQ-100 datasets do not have the same number of rows.")

    if not sp500_df["date"].equals(nasdaq_df["date"]):
        raise ValueError("S&P 500 and NASDAQ-100 dates are not aligned.")

    if not sp500_df["date"].is_monotonic_increasing:
        raise ValueError("S&P 500 dates are not sorted oldest to newest.")

    if not nasdaq_df["date"].is_monotonic_increasing:
        raise ValueError("NASDAQ-100 dates are not sorted oldest to newest.")

    required_cols = ["date", "close", "vix_close"]

    for col in required_cols:
        if col not in sp500_df.columns:
            raise ValueError(f"S&P 500 missing required column: {col}")

        if col not in nasdaq_df.columns:
            raise ValueError(f"NASDAQ-100 missing required column: {col}")

    if sp500_df[required_cols].isna().any().any():
        raise ValueError("S&P 500 processed dataset contains missing values in required columns.")

    if nasdaq_df[required_cols].isna().any().any():
        raise ValueError("NASDAQ-100 processed dataset contains missing values in required columns.")

    print("Validation passed.")
    print(f"Rows: {len(sp500_df)}")
    print(f"Start date: {sp500_df['date'].min().date()}")
    print(f"End date: {sp500_df['date'].max().date()}")


def preprocess_all():
    """
    Main preprocessing pipeline.

    Input:
    - data/raw/sp500_futures.csv
    - data/raw/nasdaq100_futures.csv
    - data/raw/vix.csv

    Output:
    - data/processed/sp500_merged.csv
    - data/processed/nasdaq100_merged.csv
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    sp500_raw = load_raw_csv(SP500_RAW)
    nasdaq_raw = load_raw_csv(NASDAQ_RAW)
    vix_raw = load_raw_csv(VIX_RAW)

    print("Standardizing columns and sorting oldest to newest...")
    sp500 = standardize_market_data(sp500_raw, is_vix=False)
    nasdaq = standardize_market_data(nasdaq_raw, is_vix=False)
    vix = standardize_market_data(vix_raw, is_vix=True)

    print("Filtering date range...")
    sp500 = filter_date_range(sp500)
    nasdaq = filter_date_range(nasdaq)
    vix = filter_date_range(vix)

    print("Aligning dates across S&P 500, NASDAQ-100, and VIX...")
    sp500, nasdaq, vix = align_all_dates(sp500, nasdaq, vix)

    print("Merging VIX into futures datasets...")
    sp500_merged = merge_with_vix(sp500, vix)
    nasdaq_merged = merge_with_vix(nasdaq, vix)

    print("Dropping rows with missing required values...")
    required_cols = ["date", "close", "vix_close"]

    sp500_merged = sp500_merged.dropna(subset=required_cols).reset_index(drop=True)
    nasdaq_merged = nasdaq_merged.dropna(subset=required_cols).reset_index(drop=True)

    print("Re-aligning dates after dropping missing required values...")
    common_dates = set(sp500_merged["date"]).intersection(set(nasdaq_merged["date"]))
    sp500_merged = (sp500_merged[sp500_merged["date"].isin(common_dates)].sort_values("date").reset_index(drop=True))
    nasdaq_merged = (nasdaq_merged[nasdaq_merged["date"].isin(common_dates)].sort_values("date").reset_index(drop=True))

    print("Validating processed datasets...")
    validate_processed_data(sp500_merged, nasdaq_merged)

    print("Saving processed files...")
    sp500_merged.to_csv(SP500_OUTPUT, index=False)
    nasdaq_merged.to_csv(NASDAQ_OUTPUT, index=False)

    print(f"Saved: {SP500_OUTPUT}")
    print(f"Saved: {NASDAQ_OUTPUT}")

    return sp500_merged, nasdaq_merged


if __name__ == "__main__":
    preprocess_all()