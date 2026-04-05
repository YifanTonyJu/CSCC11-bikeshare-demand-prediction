import pandas as pd
import numpy as np


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the processed dataset.
    """
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with missing values.
    """
    return df.dropna().copy()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    First 80% for training, last 20% for testing.
    """
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target.
    Also convert boolean columns to integers.
    """
    X = df.drop(columns=["trips"]).copy()
    y = df["trips"].copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    return X, y


def standardize_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize training and test features using training mean/std.
    Returns numpy arrays plus training mean/std for reference.
    """
    train_mean = X_train.mean(axis=0).to_numpy()
    train_std = X_train.std(axis=0).replace(0, 1).to_numpy()

    X_train_scaled = (X_train.to_numpy(dtype=float) - train_mean) / train_std
    X_test_scaled = (X_test.to_numpy(dtype=float) - train_mean) / train_std

    return X_train_scaled, X_test_scaled, train_mean, train_std


class TwoLayerNN:
    """
    Two-layer neural network for regression.

    Architecture:
        input -> hidden (ReLU) -> output (linear)

    Shapes:
        X:  (n_samples, input_dim)
        W1: (input_dim, hidden_dim)
        b1: (1, hidden_dim)
        W2: (hidden_dim, 1)
        b2: (1, 1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        learning_rate: float = 0.001,
        seed: int = 42,
    ) -> None:
        np.random.seed(seed)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # He initialization is good for ReLU
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    @staticmethod
    def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Forward pass.
        """
        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)

        Z2 = A1 @ self.W2 + self.b2
        y_pred = Z2  # linear output for regression

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
        }
        return y_pred, cache

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray, cache: dict) -> None:
        """
        Backpropagation and parameter update.
        """
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]

        n = X.shape[0]

        # dL/dy_pred for MSE = 2*(y_pred - y_true)/n
        dZ2 = (2.0 / n) * (y_pred - y_true)              # (n, 1)
        dW2 = A1.T @ dZ2                                 # (hidden_dim, 1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)         # (1, 1)

        dA1 = dZ2 @ self.W2.T                            # (n, hidden_dim)
        dZ1 = dA1 * self.relu_derivative(Z1)             # (n, hidden_dim)
        dW1 = X.T @ dZ1                                  # (input_dim, hidden_dim)
        db1 = np.sum(dZ1, axis=0, keepdims=True)         # (1, hidden_dim)

        # Gradient descent update
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 1000,
        print_every: int = 100,
    ) -> list[float]:
        """
        Train the network using full-batch gradient descent.
        """
        losses = []

        for epoch in range(1, epochs + 1):
            y_pred, cache = self.forward(X_train)
            loss = self.mse_loss(y_train, y_pred)
            losses.append(loss)

            self.backward(y_train, y_pred, cache)

            if epoch % print_every == 0 or epoch == 1:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict outputs for X.
        """
        y_pred, _ = self.forward(X)
        return y_pred


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, MSE, RMSE.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
    }


def main() -> None:
    file_path = r"D:\HuaweiMoveData\Users\huaxi\Desktop\Winter semester 3\CSCC11\project\final_dataset.csv"

    # Load and prepare data
    df = load_data(file_path)
    df = clean_data(df)

    train_df, test_df = split_data(df)

    X_train_df, y_train_s = prepare_features(train_df)
    X_test_df, y_test_s = prepare_features(test_df)

    # Standardize X
    X_train, X_test, _, _ = standardize_data(X_train_df, X_test_df)

    # Convert y to column vectors
    y_train = y_train_s.to_numpy(dtype=float).reshape(-1, 1)
    y_test = y_test_s.to_numpy(dtype=float).reshape(-1, 1)

    # Build model
    model = TwoLayerNN(
        input_dim=X_train.shape[1],
        hidden_dim=16,
        learning_rate=0.001,
        seed=42,
    )

    # Train
    model.train(X_train, y_train, epochs=2000, print_every=200)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate_regression(y_test, y_pred)

    print("\nTwo-Layer Neural Network Results")
    print(f"MAE:  {results['MAE']:.4f}")
    print(f"MSE:  {results['MSE']:.4f}")
    print(f"RMSE: {results['RMSE']:.4f}")


if __name__ == "__main__":
    main()