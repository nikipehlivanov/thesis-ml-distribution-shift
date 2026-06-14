import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# MLP model configuration

MLP_FEATURES = [
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

# Validation helpers

def validate_input_data(train_df, test_df):
    """
    Validate that train_df and test_df contain all required MLP features
    and that there are no missing values.
    """
    required_train_columns = MLP_FEATURES + [TARGET_COLUMN]
    required_test_columns = MLP_FEATURES

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
            "train_df contains missing values in MLP features or target."
        )

    if test_df[required_test_columns].isna().any().any():
        raise ValueError(
            "test_df contains missing values in MLP features."
        )

    if len(train_df) == 0:
        raise ValueError("train_df is empty.")

    if len(test_df) == 0:
        raise ValueError("test_df is empty.")

# Data preparation

def prepare_mlp_data(train_df, test_df):
    """
    Prepare X_train, y_train, and X_test for the MLP model.

    MLP uses the full feature set:
    - lagged returns
    - rolling volatility
    - lagged VIX
    - volume change

    Important:
    Neural networks require feature scaling.
    Scaling should be fitted only on the training data.
    """
    validate_input_data(train_df, test_df)

    X_train = train_df[MLP_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    X_test = test_df[MLP_FEATURES].copy()

    return X_train, y_train, X_test

# Model definition

def build_mlp_model():
    """
    Build a small feed-forward neural network.

    The model should be intentionally simple to keep model complexity
    comparable with the statistical and other ML models.

    Suggested architecture:
    - hidden_layer_sizes=(16, 8)
    - activation="relu"
    - solver="adam"
    - alpha for L2 regularization
    - max_iter sufficiently high
    - random_state fixed
    """
    base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(
        hidden_layer_sizes=(8,),
        activation="relu",
        solver="adam",
        alpha=0.01,
        learning_rate_init=0.0001,
        max_iter=5000,
        random_state=RANDOM_STATE))])

    model = TransformedTargetRegressor(
        regressor=base_model,
        transformer=StandardScaler()
    )

    return model

# Model fitting and prediction

def fit_mlp_model(train_df):
    """
    Fit the MLP model on the training data.

    The model should be fitted on the stable training period only.
    """
    validate_input_data(train_df, train_df)

    X_train = train_df[MLP_FEATURES].copy()
    y_train = train_df[TARGET_COLUMN].copy()

    model = build_mlp_model()

    model.fit(X_train, y_train)

    return model

def predict_mlp_model(fitted_model, test_df):
    """
    Generate predictions from a fitted MLP model.
    """
    missing_columns = [
        col for col in MLP_FEATURES
        if col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in test_df: {missing_columns}"
        )

    if test_df[MLP_FEATURES].isna().any().any():
        raise ValueError(
            "test_df contains missing values in MLP features."
        )

    X_test = test_df[MLP_FEATURES].copy()

    predictions = fitted_model.predict(X_test)

    return np.asarray(predictions)


def train_and_predict(train_df, test_df):
    """
    Main function used by experiments.py.

    Steps:
    1. Validate input data.
    2. Fit MLP on train_df.
    3. Predict on test_df.
    4. Return predictions as a NumPy array.

    Returns:
        numpy.ndarray
    """
    validate_input_data(train_df, test_df)
    fitted_model = fit_mlp_model(train_df)
    predictions = predict_mlp_model(fitted_model, test_df)

    if len(predictions) != len(test_df):
        raise ValueError(
            "Prediction length does not match test_df length."
        )
    
    return predictions

# Optional diagnostic helper

def get_model_config():
    """
    Return the MLP model configuration.

    Useful later for reporting hyperparameters in the thesis appendix.
    """
    return {
        "model": "MLPRegressor",
        "features": MLP_FEATURES,
        "target": TARGET_COLUMN,
        "hidden_layer_sizes": (16, 8),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.001,
        "max_iter": 2000,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
        "scaling": "StandardScaler fitted on training data only",
    }