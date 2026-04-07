import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
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
    """
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    """
    X = df.drop(columns=["trips"])
    y = df["trips"]
    return X, y


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the model using MAE, MSE, and RMSE.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse
    }


def main() -> None:
    file_path = "data/processed/final_dataset.csv"

    # Load and clean data
    df = load_data(file_path)
    df = clean_data(df)

    # Split data
    train_df, test_df = split_data(df)

    # Prepare features
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    # Train model
    model = train_linear_regression(X_train, y_train)

    # Evaluate
    results = evaluate_model(model, X_test, y_test)

    print("Linear Regression Results")
    print(f"MAE:  {results['MAE']:.4f}")
    print(f"MSE:  {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")


if __name__ == "__main__":
    main()