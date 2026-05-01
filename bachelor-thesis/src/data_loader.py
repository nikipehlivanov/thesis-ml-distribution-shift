from pathlib import Path
import pandas as pd

def parse_investing_number(value):
    """
    Convert Investing.com-style numeric strings into floats.

    Parameters
    value : any
    - Raw value from the CSV file.

    Returns
    float or pd.NA
    - Clean numeric value.
    """
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()
    value = value.replace('"', "")
    value = value.replace("%", "")

    if value in ["", "-", "nan", "NaN", "None"]:
        return pd.NA

    multiplier = 1.0

    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000.0
        value = value[:-1]

    value = value.replace(",", "")

    try:
        return float(value) * multiplier
    except ValueError:
        return pd.NA


def standardize_column_names(df):
    """
    Clean column names by stripping spaces and removing quotes.

    Parameters
    - df : pandas.DataFrame

    Returns
    - pandas.DataFrame
    """
    df = df.copy()
    df.columns = [str(col).strip().replace('"', "") for col in df.columns]
    return df


def parse_date_column(date_series):
    """
    Convert a date column to pandas datetime.

    Investing.com usually uses MM/DD/YYYY.
    FRED often uses YYYY-MM-DD.

    This function handles both.

    Parameters
    - date_series : pandas.Series

    Returns
    - pandas.Series
    """
    parsed = pd.to_datetime(date_series, format="%m/%d/%Y", errors="coerce")

    # If the MM/DD/YYYY parsing failed for many rows, try automatic parsing
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(date_series, errors="coerce")

    return parsed

def load_investing_futures_csv(path):
    """
    Load a futures CSV file downloaded from Investing.com.

    Expected raw columns:
        Date, Price, Open, High, Low, Vol., Change %

    The loader converts them into:
        date, close, open, high, low, volume

    Notes
    - Price is treated as the closing price.
    - Change % is ignored because returns are calculated later from prices.
    - Data is sorted oldest to newest.

    Parameters
    - path : str or pathlib.Path
        Path to the raw futures CSV file.

    Returns
    - pandas.DataFrame
        Clean futures dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = standardize_column_names(df)

    required_cols = ["Date", "Price", "Open", "High", "Low"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing_cols}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    rename_map = {
        "Date": "date",
        "Price": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Vol.": "volume",
        "Vol": "volume",
        "Volume": "volume",
    }

    df = df.rename(columns=rename_map)

    keep_cols = ["date", "close", "open", "high", "low"]

    if "volume" in df.columns:
        keep_cols.append("volume")

    df = df[keep_cols].copy()

    df["date"] = parse_date_column(df["date"])

    numeric_cols = ["close", "open", "high", "low"]

    if "volume" in df.columns:
        numeric_cols.append("volume")

    for col in numeric_cols:
        df[col] = df[col].apply(parse_investing_number)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    return df

def load_vix_csv(path):
    """
    Load VIX data.
    Supports two formats:
    1. FRED-style format:
        Date, Close
    2. Investing.com-style format:
        Date, Price, Open, High, Low, Vol., Change %
    Output columns:
        date, vix_close

    Parameters
    - path : str or pathlib.Path
        Path to VIX CSV file.

    Returns
    - pandas.DataFrame
        Clean VIX dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = standardize_column_names(df)

    if "Date" not in df.columns:
        raise ValueError(
            f"Missing Date column in {path.name}.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    if "Price" in df.columns:
        value_col = "Price"
    elif "Close" in df.columns:
        value_col = "Close"
    elif "VIXCLS" in df.columns:
        value_col = "VIXCLS"
    else:
        raise ValueError(
            f"Could not find a VIX value column in {path.name}.\n"
            f"Expected one of: Price, Close, VIXCLS.\n"
            f"Columns found: {df.columns.tolist()}"
        )

    df = df[["Date", value_col]].copy()
    df = df.rename(columns={"Date": "date", value_col: "vix_close"})

    df["date"] = parse_date_column(df["date"])

    df["vix_close"] = df["vix_close"].apply(parse_investing_number)
    df["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce")

    df = df.dropna(subset=["date", "vix_close"])
    df = df.drop_duplicates(subset=["date"], keep="last")

    df = df.sort_values("date").reset_index(drop=True)

    return df

def merge_with_vix(asset_df, vix_df):
    """
    Merge an asset futures dataframe with VIX data by date.

    Parameters
    - asset_df : pandas.DataFrame
        Clean futures dataframe with a date column.

    - vix_df : pandas.DataFrame
        Clean VIX dataframe with columns date and vix_close.

    Returns
    - pandas.DataFrame
        Merged dataframe containing futures data and VIX.
    """
    if "date" not in asset_df.columns:
        raise ValueError("asset_df must contain a 'date' column.")

    if "date" not in vix_df.columns:
        raise ValueError("vix_df must contain a 'date' column.")

    if "vix_close" not in vix_df.columns:
        raise ValueError("vix_df must contain a 'vix_close' column.")

    merged = asset_df.merge(vix_df[["date", "vix_close"]], on="date", how="inner")
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged

def load_project_raw_data(raw_dir):
    """
    Load all three raw datasets used in the thesis project.

    Expected files:
        raw_dir/sp500_futures.csv
        raw_dir/nasdaq100_futures.csv
        raw_dir/vix.csv

    Parameters
    - raw_dir : str or pathlib.Path
        Path to data/raw directory.
    Returns

    dict
        Dictionary containing:
            sp500
            nasdaq100
            vix
            sp500_merged
            nasdaq100_merged
    """
    raw_dir = Path(raw_dir)

    sp500_path = raw_dir / "sp500_futures.csv"
    nasdaq_path = raw_dir / "nasdaq100_futures.csv"
    vix_path = raw_dir / "vix.csv"

    sp500 = load_investing_futures_csv(sp500_path)
    nasdaq100 = load_investing_futures_csv(nasdaq_path)
    vix = load_vix_csv(vix_path)

    sp500_merged = merge_with_vix(sp500, vix)
    nasdaq100_merged = merge_with_vix(nasdaq100, vix)

    return {
        "sp500": sp500,
        "nasdaq100": nasdaq100,
        "vix": vix,
        "sp500_merged": sp500_merged,
        "nasdaq100_merged": nasdaq100_merged,
    }


project_root = Path(__file__).resolve().parents[1]
raw_dir = project_root / "data" / "raw"

data = load_project_raw_data(raw_dir)

for name, df in data.items():
    print("=" * 80)
    print(name)
    print("=" * 80)
    print(df.head())
    print()
    print(df.tail())
    print()
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Missing values:")
    print(df.isna().sum())
    print()