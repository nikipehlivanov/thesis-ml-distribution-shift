import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Ridge model configuration

RIDGE_FEATURES = [
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

RIDGE_ALPHA = 1.0

# Validation helpers

def validate_input_data(train_df, test_df):
    """
    Validate that train_df and test_df contain all required Ridge features
    and that there are no missing values.
    """
    required_train_columns = RIDGE_FEATURES + [TARGET_COLUMN]
    required_test_columns = RIDGE_FEATURES

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
            "train_df contains missing values in Ridge features or target."
        )

    if test_df[required_test_columns].isna().any().any():
        raise ValueError(
            "test_df contains missing values in Ridge features."
        )

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(test_df) == 0:
        raise ValueError("test_df is empty.")
    
# Data preparation

def prepare_ridge_data(train_df, test_df):
    """
    Prepare X_train, y_train, and X_test for the Ridge model.

    Ridge uses the full feature set:
    - lagged returns
    - rolling volatility
    - lagged VIX
    - volume change

    Important:
    Ridge Regression should use scaled features because L2 regularization
    is sensitive to feature scale.
    """
    validate_input_data(train_df, test_df)

    X_train = train_df[RIDGE_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[RIDGE_FEATURES].copy()

    return X_train, y_train, X_test

# Model definition

def build_ridge_model():
    """
    Build the Ridge Regression pipeline.

    Pipeline:
    - StandardScaler: fitted only on training data
    - Ridge: L2-regularized linear regression

    Suggested:
    - alpha = RIDGE_ALPHA
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(
           alpha= RIDGE_ALPHA 
        ))
    ])

    return model

# Model fitting and prediction

def fit_ridge_model(train_df):
    """
    Fit the Ridge model on the training data.

    The scaler should be fitted only on the training data through the Pipeline.
    """
    validate_input_data(train_df, train_df)

    X_train = train_df[RIDGE_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    model = build_ridge_model()

    model.fit(X_train, y_train)

    return model

def predict_ridge_model(fitted_model, test_df):
    """
    Generate predictions from a fitted Ridge model.
    """
    missing_columns = [
        col for col in RIDGE_FEATURES
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_columns}"
        )

    if test_df[RIDGE_FEATURES].isna().any().any():
        raise ValueError(
            "test_df contains missing values in Ridge features."
        )

    X_test = test_df[RIDGE_FEATURES].copy()

    predictions = fitted_model.predict(X_test)

    return np.asarray(predictions)


def train_and_predict(train_df, test_df):
    """
    Main function used by experiments.py.

    Steps:
    1. Validate input data.
    2. Fit Ridge on train_df.
    3. Predict on test_df.
    4. Return predictions as a NumPy array.

    Returns:
        numpy.ndarray
    """
    validate_input_data(train_df, test_df)
    fitted_model = fit_ridge_model(train_df)
    predictions = predict_ridge_model(fitted_model, test_df)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction length does not match test_df length."
        )
    
    return predictions

# Optional diagnostic helper

def get_model_config():
    """
    Return the Ridge model configuration.

    Useful later for reporting hyperparameters in the thesis appendix.
    """
    return {
        "model": "Ridge Regression",
        "features": RIDGE_FEATURES,
        "target": TARGET_COLUMN,
        "alpha": RIDGE_ALPHA,
        "regularization": "L2",
        "scaling": "StandardScaler fitted on training data only",
    }