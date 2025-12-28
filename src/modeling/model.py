import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Paths
FEATURE_DATA_PATH = "data/processed/features.csv"
MODEL_OUTPUT_PATH = "data/models/model_results.csv"


def train_models(input_path: str, output_path: str):
    # Load feature-engineered data
    df = pd.read_csv(input_path)

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    # Model 1: Logistic Regression (Classical ML)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    results.append({
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, lr_preds)
    })

    # Model 2: Random Forest (Classical ML)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append({
        "model": "Random Forest",
        "accuracy": accuracy_score(y_test, rf_preds)
    })

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print("Model training completed. Results saved.")


if __name__ == "__main__":
    train_models(FEATURE_DATA_PATH, MODEL_OUTPUT_PATH)
