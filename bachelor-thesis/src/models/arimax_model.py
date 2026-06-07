import numpy as np
import statsmodels.api as sm

# ARIMAX model configuration

ARIMAX_FEATURES = [
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

# Validation helpers

def validate_input_data(train_df, test_df):
    """
    Validate that the required ARIMAX columns exist and contain no missing values.
    """

    required_train_columns = ARIMAX_FEATURES + [TARGET_COLUMN]
    required_test_columns = ARIMAX_FEATURES

    missing_train_columns = [
        col for col in required_train_columns
        if col not in train_df.columns
    ]

    missing_test_columns = [
        col for col in required_test_columns
        if col not in test_df.columns
    ]

    if missing_train_columns:
        raise ValueError(
            f"Missing required columns in train_df: {missing_train_columns}"
        )

    if missing_test_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_test_columns}"
        )

    if train_df[required_train_columns].isna().any().any():
        raise ValueError(
            "train_df contains missing values in ARIMAX features or target."
        )

    if test_df[required_test_columns].isna().any().any():
        raise ValueError(
            "test_df contains missing values in ARIMAX features."
        )

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(test_df) == 0:
        raise ValueError("test_df is empty.")


# Data preparation

def prepare_arimax_data(train_df, test_df):
    """
    Prepare training and testing matrices for the ARIMAX model.

    ARIMAX uses:
    - lagged returns
    - volatility features
    - lagged VIX
    - volume change

    Target:
    - next-day log return
    """

    validate_input_data(train_df, test_df)

    X_train = train_df[ARIMAX_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[ARIMAX_FEATURES].copy()

    # Add intercept/constant term
    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    return X_train, y_train, X_test

# Model fitting and prediction

def fit_arimax_model(train_df):
    """
    Fit an ARIMAX-style statistical model.

    Model form:
        target_t = c
                   + b1 * ret_lag_1
                   + ...
                   + b10 * ret_lag_10
                   + b11 * volatility_5
                   + b12 * volatility_10
                   + b13 * vix_lag_1
                   + b14 * volume_change
                   + error_t

    In this implementation, the model is estimated as an OLS regression
    with autoregressive lag features and exogenous predictors.
    """

    required_columns = ARIMAX_FEATURES + [TARGET_COLUMN]

    missing_columns = [
        col for col in required_columns
        if col not in train_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in train_df: {missing_columns}"
        )

    if train_df[required_columns].isna().any().any():
        raise ValueError(
            "train_df contains missing values in ARIMAX features or target."
        )

    X_train = train_df[ARIMAX_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_train = sm.add_constant(X_train, has_constant="add")

    model = sm.OLS(y_train, X_train)
    fitted_model = model.fit()

    return fitted_model


def predict_arimax_model(fitted_model, test_df):
    """
    Generate predictions from a fitted ARIMAX model.
    """

    missing_columns = [
        col for col in ARIMAX_FEATURES
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_columns}"
        )

    if test_df[ARIMAX_FEATURES].isna().any().any():
        raise ValueError(
            "test_df contains missing values in ARIMAX features."
        )

    X_test = test_df[ARIMAX_FEATURES].copy()
    X_test = sm.add_constant(X_test, has_constant="add")

    predictions = fitted_model.predict(X_test)

    return np.asarray(predictions)


def train_and_predict(train_df, test_df):
    """
    Fit the ARIMAX model on training data and predict on test data.

    This is the main function that experiments.py will call.

    Parameters
    ----------
    train_df : pandas DataFrame
        Training data, e.g. 2010-2017 stable period.

    test_df : pandas DataFrame
        Test data, e.g. 2018-2019 ID period or 2020-2021 OOD period.

    Returns
    -------
    numpy.ndarray
        Predicted next-day log returns for the test period.
    """

    validate_input_data(train_df, test_df)

    fitted_model = fit_arimax_model(train_df)
    predictions = predict_arimax_model(fitted_model, test_df)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction length does not match test_df length."
        )

    return predictions

def get_model_summary(train_df):
    """
    Fit the ARIMAX model and return the statsmodels summary.

    Useful for thesis interpretation, because it shows:
    - estimated coefficients
    - p-values
    - R-squared
    - which exogenous variables contribute most
    """

    fitted_model = fit_arimax_model(train_df)
    return fitted_model.summary()