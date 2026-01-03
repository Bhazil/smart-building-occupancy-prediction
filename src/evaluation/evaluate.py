import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Get project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

# Paths
DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_features.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "occupancy_model.pkl"
)

def evaluate_model():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Drop non-numeric columns (date)
    df = df.select_dtypes(exclude=["object"])

    X = df.drop(columns=["occupancy"])
    y = df["occupancy"]

    # Train-test split (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load model
    model = joblib.load(MODEL_PATH)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Evaluation Results")
    print("------------------")
    print("Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()
