import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Random Forest model configuration

RF_FEATURES = [
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

RANDOM_STATE = 42

N_ESTIMATORS = 300
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 20

# Validation helpers

def validate_input_data(train_df, test_df):
    """
    Validate that train_df and test_df contain all required Random Forest
    features and that there are no missing values.
    """
    required_train_columns = RF_FEATURES + [TARGET_COLUMN]
    required_test_columns = RF_FEATURES

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
            "train_df contains missing values in RF features or target."
        )

    if test_df[required_test_columns].isna().any().any():
        raise ValueError(
            "test_df contains missing values in RF features."
        )

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(test_df) == 0:
        raise ValueError("test_df is empty.")

# Data preparation

def prepare_rf_data(train_df, test_df):
    """
    Prepare X_train, y_train, and X_test for the Random Forest model.

    Random Forest uses the full feature set:
    - lagged returns
    - rolling volatility
    - lagged VIX
    - volume change

    Important:
    Random Forest does not require feature scaling.
    """
    validate_input_data(train_df, test_df)

    X_train = train_df[RF_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[RF_FEATURES].copy()

    return X_train, y_train, X_test

# Model definition

def build_random_forest_model():
    """
    Build a moderate-complexity Random Forest model.

    Important thesis design choice:
    The model complexity is intentionally restricted to keep the comparison
    fair against statistical models and other moderate ML models.

    Suggested controls:
    - n_estimators = 300
    - max_depth = 5
    - min_samples_leaf = 20
    - random_state fixed
    """
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return model

# Model fitting and prediction

def fit_random_forest_model(train_df):
    """
    Fit the Random Forest model on the training data.

    The model should be fitted only on the stable training period.
    """
    validate_input_data(train_df, train_df)

    X_train = train_df[RF_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    model = build_random_forest_model()

    model.fit(X_train, y_train)

    return model


def predict_random_forest_model(fitted_model, test_df):
    """
    Generate predictions from a fitted Random Forest model.
    """
    missing_columns = [
        col for col in RF_FEATURES
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_columns}"
        )

    if test_df[RF_FEATURES].isna().any().any():
        raise ValueError(
            "test_df contains missing values in RF features."
        )

    X_test = test_df[RF_FEATURES].copy()

    predictions = fitted_model.predict(X_test)

    return np.asarray(predictions)


def train_and_predict(train_df, test_df):
    """
    Main function used by experiments.py.

    Steps:
    1. Validate input data.
    2. Fit Random Forest on train_df.
    3. Predict on test_df.
    4. Return predictions as a NumPy array.

    Returns:
        numpy.ndarray
    """
    validate_input_data(train_df, test_df)
    fitted_model = fit_random_forest_model(train_df)
    predictions = predict_random_forest_model(fitted_model, test_df)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction length does not match test_df length."
        )
    
    return predictions

# Optional diagnostic helper

def get_model_config():
    """
    Return the Random Forest model configuration.

    Useful later for reporting hyperparameters in the thesis appendix.
    """
    return {
        "model": "Random Forest Regressor",
        "features": RF_FEATURES,
        "target": TARGET_COLUMN,
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "random_state": RANDOM_STATE,
        "scaling": "Not used; tree-based models do not require feature scaling",
        "complexity_control": (
            "Depth and minimum leaf size are restricted to keep the model "
            "moderate in complexity and comparable with the other models."
        ),
    }