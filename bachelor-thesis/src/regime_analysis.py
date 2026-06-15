from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis

# Project paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Configuration

ASSET_SPLITS = {
    "sp500": {
        "id_test": SPLITS_DIR / "sp500_id_test.csv",
        "ood_test": SPLITS_DIR / "sp500_ood_test.csv",
    },
    "nasdaq100": {
        "id_test": SPLITS_DIR / "nasdaq100_id_test.csv",
        "ood_test": SPLITS_DIR / "nasdaq100_ood_test.csv",
    },
}

PERIOD_LABELS = {
    "id_test": "Stable period: 2018-2019",
    "ood_test": "Crisis/OOD period: 2020-2021",
}

RETURN_COLUMN = "log_return"
TARGET_COLUMN = "target"

VIX_COLUMNS = [
    "vix_close",
    "vix_lag_1",
]

# Loading helpers

def load_split(path):
    """
    Load a split file and sort it by date.
    """
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {path}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        raise ValueError(f"Invalid dates found in {path}")

    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_return_column(df):
    """
    Prefer log_return for regime analysis.

    If log_return is not available, fall back to target.
    """
    if RETURN_COLUMN in df.columns:
        return RETURN_COLUMN

    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN

    raise ValueError(
        f"Neither {RETURN_COLUMN} nor {TARGET_COLUMN} found in dataframe."
    )


def get_vix_column(df):
    """
    Find available VIX column.
    """
    for col in VIX_COLUMNS:
        if col in df.columns:
            return col

    return None

# Summary statistics

def compute_regime_statistics(asset_name, period_name, df):
    """
    Compute distribution and regime statistics for one asset and one period.
    """
    return_col = get_return_column(df)
    vix_col = get_vix_column(df)

    returns = df[return_col].dropna()

    if returns.empty:
        raise ValueError(f"No valid returns found for {asset_name}, {period_name}")

    stats = {
        "asset": asset_name,
        "period": period_name,
        "period_label": PERIOD_LABELS[period_name],
        "n_observations": len(returns),
        "mean_return": returns.mean(),
        "std_return": returns.std(),
        "mean_abs_return": returns.abs().mean(),
        "min_return": returns.min(),
        "max_return": returns.max(),
        "skewness": skew(returns),
        "kurtosis": kurtosis(returns, fisher=True),
    }

    if vix_col is not None:
        vix_values = df[vix_col].dropna()

        stats["mean_vix"] = vix_values.mean()
        stats["median_vix"] = vix_values.median()
        stats["max_vix"] = vix_values.max()
    else:
        stats["mean_vix"] = np.nan
        stats["median_vix"] = np.nan
        stats["max_vix"] = np.nan

    return stats


def create_regime_summary_table():
    """
    Create summary statistics table for stable vs crisis periods.
    """
    rows = []

    for asset_name, split_paths in ASSET_SPLITS.items():
        for period_name, path in split_paths.items():
            df = load_split(path)
            rows.append(compute_regime_statistics(asset_name, period_name, df))

    summary_df = pd.DataFrame(rows)

    return summary_df


def create_regime_change_table(summary_df):
    """
    Compare OOD crisis period against ID stable period.

    Ratios above 1 indicate higher values in the crisis period.
    """
    rows = []

    for asset_name in summary_df["asset"].unique():
        asset_df = summary_df[summary_df["asset"] == asset_name]

        id_row = asset_df[asset_df["period"] == "id_test"].iloc[0]
        ood_row = asset_df[asset_df["period"] == "ood_test"].iloc[0]

        row = {
            "asset": asset_name,
            "std_return_id": id_row["std_return"],
            "std_return_ood": ood_row["std_return"],
            "std_return_ratio_ood_over_id": (
                ood_row["std_return"] / id_row["std_return"]
            ),
            "mean_abs_return_id": id_row["mean_abs_return"],
            "mean_abs_return_ood": ood_row["mean_abs_return"],
            "mean_abs_return_ratio_ood_over_id": (
                ood_row["mean_abs_return"] / id_row["mean_abs_return"]
            ),
            "mean_vix_id": id_row["mean_vix"],
            "mean_vix_ood": ood_row["mean_vix"],
            "mean_vix_ratio_ood_over_id": (
                ood_row["mean_vix"] / id_row["mean_vix"]
                if pd.notna(id_row["mean_vix"]) and id_row["mean_vix"] != 0
                else np.nan
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)

# Saving helpers

def save_table(df, filename):
    """
    Save table as CSV and LaTeX.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / f"{filename}.csv"
    tex_path = TABLES_DIR / f"{filename}.tex"

    df.to_csv(csv_path, index=False)

    df.to_latex(
        tex_path,
        index=False,
        float_format="%.6f"
    )

    return csv_path, tex_path


def save_figure(fig, filename):
    """
    Save figure to results/figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path

# Gaussian distribution plots

def plot_gaussian_regime_comparison(asset_name):
    """
    Plot stable vs crisis return distributions with fitted Gaussian curves.

    The Gaussian curves are used as visual approximations, not as an assumption
    that financial returns are normally distributed.
    """
    id_df = load_split(ASSET_SPLITS[asset_name]["id_test"])
    ood_df = load_split(ASSET_SPLITS[asset_name]["ood_test"])

    return_col = get_return_column(id_df)

    id_returns = id_df[return_col].dropna().values
    ood_returns = ood_df[return_col].dropna().values

    id_mu, id_sigma = norm.fit(id_returns)
    ood_mu, ood_sigma = norm.fit(ood_returns)

    x_min = min(id_returns.min(), ood_returns.min())
    x_max = max(id_returns.max(), ood_returns.max())

    x_values = np.linspace(x_min, x_max, 500)

    id_pdf = norm.pdf(x_values, id_mu, id_sigma)
    ood_pdf = norm.pdf(x_values, ood_mu, ood_sigma)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        id_returns,
        bins=40,
        density=True,
        alpha=0.45,
        label="Stable period returns"
    )

    ax.hist(
        ood_returns,
        bins=40,
        density=True,
        alpha=0.45,
        label="Crisis/OOD period returns"
    )

    ax.plot(
        x_values,
        id_pdf,
        linewidth=2,
        label=f"Stable Gaussian fit, σ={id_sigma:.4f}"
    )

    ax.plot(
        x_values,
        ood_pdf,
        linewidth=2,
        label=f"Crisis Gaussian fit, σ={ood_sigma:.4f}"
    )

    ax.axvline(0, linewidth=1)

    ax.set_title(f"{asset_name.upper()} Return Distribution: Stable vs Crisis")
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return save_figure(fig, f"{asset_name}_gaussian_regime_comparison.png")


def plot_vix_regime_comparison(asset_name):
    """
    Plot VIX distribution in stable vs crisis period.
    """
    id_df = load_split(ASSET_SPLITS[asset_name]["id_test"])
    ood_df = load_split(ASSET_SPLITS[asset_name]["ood_test"])

    vix_col = get_vix_column(id_df)

    if vix_col is None:
        print(f"No VIX column found for {asset_name}. Skipping VIX plot.")
        return None

    id_vix = id_df[vix_col].dropna()
    ood_vix = ood_df[vix_col].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(
        [id_vix, ood_vix],
        labels=["Stable\n2018-2019", "Crisis/OOD\n2020-2021"]
    )

    ax.set_title(f"{asset_name.upper()} VIX Regime Comparison")
    ax.set_ylabel("VIX")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return save_figure(fig, f"{asset_name}_vix_regime_comparison.png")


def plot_absolute_return_regime_comparison(asset_name):
    """
    Plot absolute daily returns over time for ID and OOD periods.

    This visualizes increased realized volatility during the crisis period.
    """
    id_df = load_split(ASSET_SPLITS[asset_name]["id_test"])
    ood_df = load_split(ASSET_SPLITS[asset_name]["ood_test"])

    return_col = get_return_column(id_df)

    id_df = id_df.copy()
    ood_df = ood_df.copy()

    id_df["abs_return"] = id_df[return_col].abs()
    ood_df["abs_return"] = ood_df[return_col].abs()

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        id_df["date"],
        id_df["abs_return"],
        label="Stable period absolute returns"
    )

    ax.plot(
        ood_df["date"],
        ood_df["abs_return"],
        label="Crisis/OOD period absolute returns"
    )

    ax.set_title(f"{asset_name.upper()} Absolute Returns Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Absolute daily log return")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    return save_figure(fig, f"{asset_name}_absolute_returns_over_time.png")

# Full regime analysis pipeline

def run_regime_analysis():
    """
    Run regime analysis.

    Outputs:
    - Gaussian distribution comparison plots
    - VIX regime plots
    - absolute return plots
    - regime summary tables
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Creating regime summary tables...")

    summary_df = create_regime_summary_table()
    change_df = create_regime_change_table(summary_df)

    save_table(summary_df, "regime_summary_statistics")
    save_table(change_df, "regime_change_statistics")

    print("Creating regime figures...")

    for asset_name in ASSET_SPLITS.keys():
        plot_gaussian_regime_comparison(asset_name)
        plot_vix_regime_comparison(asset_name)
        plot_absolute_return_regime_comparison(asset_name)

    print("\nRegime analysis completed successfully.")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")

    return summary_df, change_df


if __name__ == "__main__":
    run_regime_analysis()