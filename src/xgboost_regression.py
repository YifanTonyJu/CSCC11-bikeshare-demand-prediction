import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed dataset.
    """
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values.
    Missing values mainly come from lag features.
    """
    df = df.dropna().copy()
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple train/test split.
    First 80% for training, last 20% for testing.

    This keeps the original row order, which is appropriate
    for a time-based regression setting.
    """
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.

    The target column is assumed to be 'trips',
    consistent with the current project scripts.
    """
    if "trips" not in df.columns:
        raise ValueError("Target column 'trips' was not found in the dataset.")

    X = df.drop(columns=["trips"]).copy()
    y = df["trips"].copy()
    return X, y


def encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert features into a numeric format that XGBoost can use.

    Steps:
    1. Convert boolean columns to integers
    2. One-hot encode any non-numeric columns
    3. Make sure train and test have exactly the same columns
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Convert boolean columns to integers
    bool_cols_train = X_train.select_dtypes(include=["bool"]).columns
    if len(bool_cols_train) > 0:
        X_train[bool_cols_train] = X_train[bool_cols_train].astype(int)

    bool_cols_test = X_test.select_dtypes(include=["bool"]).columns
    if len(bool_cols_test) > 0:
        X_test[bool_cols_test] = X_test[bool_cols_test].astype(int)

    # Combine train and test before one-hot encoding
    # so that both end up with the same encoded columns
    train_size = len(X_train)
    combined = pd.concat([X_train, X_test], axis=0)

    # Identify non-numeric columns and one-hot encode them
    non_numeric_cols = combined.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric_cols) > 0:
        combined = pd.get_dummies(
            combined,
            columns=list(non_numeric_cols),
            drop_first=False
        )

    # Split the combined dataframe back into train and test
    X_train_encoded = combined.iloc[:train_size].copy()
    X_test_encoded = combined.iloc[train_size:].copy()

    return X_train_encoded, X_test_encoded


def train_xgboost_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> XGBRegressor:
    """
    Train an XGBoost regression model.

    Notes:
    - Tree-based models do not require feature standardization.
    - The hyperparameters below are a reasonable starting point.
    """
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="rmse"
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate the model using MAE, MSE, and RMSE.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse)
    }


def main() -> None:
    file_path = "data/processed/final_dataset.csv"

    # Load and clean the processed dataset
    df = load_data(file_path)
    df = clean_data(df)

    # Split the dataset into training and test sets
    train_df, test_df = split_data(df)

    # Separate input features and regression target
    X_train_raw, y_train = prepare_features(train_df)
    X_test_raw, y_test = prepare_features(test_df)

    # Encode features so that all columns are numeric
    X_train, X_test = encode_features(X_train_raw, X_test_raw)

    # Train the XGBoost model
    model = train_xgboost_regression(
        X_train,
        y_train,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    # Evaluate model performance
    results = evaluate_model(model, X_test, y_test)

    print("XGBoost Regression Results")
    print(f"MAE:  {results['MAE']:.4f}")
    print(f"MSE:  {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")


if __name__ == "__main__":
    main()