import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Get project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

# Paths
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_features.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models"
)

MODEL_FILE = os.path.join(
    MODEL_PATH, "occupancy_model.pkl"
)


def train_model(input_path: str):
    """
    Trains a logistic regression model on engineered occupancy features.
    """

    # Load data
    df = pd.read_csv(input_path)
    print("Dataset shape:", df.shape)

    # Drop non-numeric columns (like date)
    non_numeric_cols = df.select_dtypes(include=["object"]).columns
    print("Dropping non-numeric columns:", list(non_numeric_cols))
    df = df.drop(columns=non_numeric_cols)

    # Separate features and target
    X = df.drop(columns=["occupancy"])
    y = df["occupancy"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Model Accuracy:", accuracy)

    # Save model
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"Model saved at: {MODEL_FILE}")


if __name__ == "__main__":
    train_model(INPUT_PATH)
