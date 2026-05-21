import numpy as np
import pandas as pd
from scipy.stats import norm

# Basic regression metrics

def mean_absolute_error(y_true, y_pred):
    """
    Mean Absolute Error (MAE).

    Measures the average absolute difference between actual and predicted returns.
    Lower MAE means better forecasting accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).

    Penalizes larger forecasting errors more strongly than MAE.
    This is especially useful in crisis periods where large errors matter.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Financial forecasting metric

def mean_directional_accuracy(y_true, y_pred):
    """
    Mean Directional Accuracy (MDA).

    Measures how often the model predicts the correct return direction.

    Example:
    actual positive + predicted positive = correct
    actual negative + predicted negative = correct

    Values:
    - 0.50 means roughly random directional guessing
    - above 0.50 means better than random direction prediction
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    true_direction = np.sign(y_true)
    pred_direction = np.sign(y_pred)

    return np.mean(true_direction == pred_direction)

# Benchmark comparison metric

def out_of_sample_r2(y_true, y_pred, benchmark_pred=None):
    """
    Out-of-sample R-squared.

    Compares the model against a simple benchmark.

    By default, the benchmark is zero return:
        tomorrow's return = 0

    Interpretation:
    - R2_OOS > 0: model beats benchmark
    - R2_OOS = 0: model equals benchmark
    - R2_OOS < 0: model is worse than benchmark
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if benchmark_pred is None:
        benchmark_pred = np.zeros_like(y_true)
    else:
        benchmark_pred = np.asarray(benchmark_pred)

    model_error = np.sum((y_true - y_pred) ** 2)
    benchmark_error = np.sum((y_true - benchmark_pred) ** 2)

    if benchmark_error == 0:
        return np.nan

    return 1 - (model_error / benchmark_error)

# Robustness metric

def performance_degradation(id_error, ood_error):
    """
    Performance degradation from stable test period to crisis test period.

    Formula:
        degradation = (OOD error - ID error) / ID error

    Interpretation:
    - positive value: model performs worse in crisis period
    - smaller value: better robustness
    - larger value: weaker robustness
    """
    if id_error == 0:
        return np.nan

    return (ood_error - id_error) / id_error

# Combined evaluation function

def evaluate_predictions(y_true, y_pred):
    """
    Calculate all main metrics for one set of predictions.

    Input:
    - y_true: actual target values
    - y_pred: model predictions

    Output:
    dictionary with:
    - MAE
    - RMSE
    - MDA
    - R2_OOS
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty.")

    if np.isnan(y_true).any():
        raise ValueError("y_true contains NaN values.")

    if np.isnan(y_pred).any():
        raise ValueError("y_pred contains NaN values.")

    metrics = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MDA": mean_directional_accuracy(y_true, y_pred),
        "R2_OOS": out_of_sample_r2(y_true, y_pred),
    }

    return metrics

# ID vs OOD comparison

def compare_id_ood(id_metrics, ood_metrics):
    """
    Compare in-distribution and out-of-distribution performance.

    Input:
    - id_metrics: metrics dictionary for stable test period
    - ood_metrics: metrics dictionary for crisis test period

    Output:
    combined dictionary with ID metrics, OOD metrics,
    and degradation values for MAE and RMSE.
    """
    comparison = {}

    for metric_name, metric_value in id_metrics.items():
        comparison[f"ID_{metric_name}"] = metric_value

    for metric_name, metric_value in ood_metrics.items():
        comparison[f"OOD_{metric_name}"] = metric_value

    comparison["MAE_Degradation"] = performance_degradation(
        id_metrics["MAE"],
        ood_metrics["MAE"],
    )

    comparison["RMSE_Degradation"] = performance_degradation(
        id_metrics["RMSE"],
        ood_metrics["RMSE"],
    )

    return comparison

# Diebold-Mariano test

def diebold_mariano_test(y_true, pred_1, pred_2, loss="squared"):
    """
    Diebold-Mariano test for equal predictive accuracy.

    This compares two forecasting models.

    Null hypothesis:
        both models have equal predictive accuracy

    Parameters:
    - y_true: actual values
    - pred_1: predictions from model 1
    - pred_2: predictions from model 2
    - loss: "squared" or "absolute"

    Returns:
    - DM statistic
    - approximate two-sided p-value

    Note:
    This is a simple version suitable for one-step-ahead forecasts.
    """
    y_true = np.asarray(y_true)
    pred_1 = np.asarray(pred_1)
    pred_2 = np.asarray(pred_2)

    if not (len(y_true) == len(pred_1) == len(pred_2)):
        raise ValueError("y_true, pred_1, and pred_2 must have the same length.")

    e1 = y_true - pred_1
    e2 = y_true - pred_2

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss must be either 'squared' or 'absolute'.")

    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    if var_d == 0:
        return {
            "DM_statistic": np.nan,
            "p_value": np.nan,
        }

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))

    return {
        "DM_statistic": dm_stat,
        "p_value": p_value,
    }

# Utility for saving metrics

def metrics_to_dataframe(results):
    """
    Convert a list of metric dictionaries into a pandas DataFrame.

    Useful later in experiments.py.

    Example input:
    [
        {"asset": "sp500", "model": "ridge", "ID_MAE": ..., "OOD_MAE": ...},
        {"asset": "nasdaq100", "model": "ridge", "ID_MAE": ..., "OOD_MAE": ...}
    ]
    """
    return pd.DataFrame(results)

# test block

if __name__ == "__main__":
    y_true = np.array([0.01, -0.02, 0.005, 0.00, 0.015])
    y_pred = np.array([0.008, -0.015, 0.004, 0.001, 0.010])

    print("Example evaluation:")
    print(evaluate_predictions(y_true, y_pred))

    id_metrics = evaluate_predictions(y_true, y_pred)
    ood_metrics = evaluate_predictions(
        y_true,
        np.array([0.02, 0.01, -0.003, 0.004, -0.01])
    )

    print("\nExample ID vs OOD comparison:")
    print(compare_id_ood(id_metrics, ood_metrics))