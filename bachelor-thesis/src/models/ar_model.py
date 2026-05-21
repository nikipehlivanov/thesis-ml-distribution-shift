import numpy as np
import statsmodels.api as sm

# AR model configuration

AR_FEATURES = [
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
]

TARGET_COLUMN = "target"

# Validation helpers

def validate_input_data(train_df, test_df):
    """
    Validate that the required AR columns exist and contain no missing values.
    """

    required_train_columns = AR_FEATURES + [TARGET_COLUMN]
    required_test_columns = AR_FEATURES

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
            "train_df contains missing values in AR features or target."
        )

    if test_df[required_test_columns].isna().any().any():
        raise ValueError(
            "test_df contains missing values in AR features."
        )

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(test_df) == 0:
        raise ValueError("test_df is empty.")

# Data preparation

def prepare_ar_data(train_df, test_df):
    """
    Prepare training and testing matrices for the AR model.

    AR uses only lagged returns:
        ret_lag_1, ..., ret_lag_10

    Target:
        next-day log return
    """

    validate_input_data(train_df, test_df)

    X_train = train_df[AR_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[AR_FEATURES].copy()

    # Add intercept term
    X_train = sm.add_constant(X_train, has_constant="add")
    X_test = sm.add_constant(X_test, has_constant="add")

    return X_train, y_train, X_test

# Model fitting and prediction

def fit_ar_model(train_df):
    """
    Fit an autoregressive-style statistical model.

    Model form:
        target_t = c
                   + b1 * ret_lag_1
                   + b2 * ret_lag_2
                   + ...
                   + b10 * ret_lag_10
                   + error_t

    Although implemented as OLS regression, this corresponds to an
    autoregressive model because only lagged returns are used as predictors.
    """

    required_columns = AR_FEATURES + [TARGET_COLUMN]

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
            "train_df contains missing values in AR features or target."
        )

    X_train = train_df[AR_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_train = sm.add_constant(X_train, has_constant="add")

    model = sm.OLS(y_train, X_train)
    fitted_model = model.fit()

    return fitted_model

def predict_ar_model(fitted_model, test_df):
    """
    Generate predictions from a fitted AR model.
    """

    missing_columns = [
        col for col in AR_FEATURES
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_columns}"
        )

    if test_df[AR_FEATURES].isna().any().any():
        raise ValueError(
            "test_df contains missing values in AR features."
        )

    X_test = test_df[AR_FEATURES].copy()
    X_test = sm.add_constant(X_test, has_constant="add")

    predictions = fitted_model.predict(X_test)

    return np.asarray(predictions)

def train_and_predict(train_df, test_df):
    """
    Fit the AR model on training data and predict on test data.

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

    fitted_model = fit_ar_model(train_df)
    predictions = predict_ar_model(fitted_model, test_df)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction length does not match test_df length."
        )

    return predictions

def get_model_summary(train_df):
    """
    Fit the AR model and return the statsmodels summary.

    Useful for thesis interpretation, because it shows:
    - estimated coefficients
    - p-values
    - R-squared
    - residual diagnostics

    This function is optional and should mainly be used for analysis,
    not for the main experiment pipeline.
    """

    fitted_model = fit_ar_model(train_df)
    return fitted_model.summary()