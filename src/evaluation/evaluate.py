import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Paths
FEATURE_DATA_PATH = "data/processed/features.csv"
EVALUATION_OUTPUT_PATH = "tables/RQ1_model_evaluation.csv"


def evaluate_models(input_path: str, output_path: str):
    # Load data
    df = pd.read_csv(input_path)

    # Split features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted"),
            "Recall": recall_score(y_test, preds, average="weighted"),
            "F1_Score": f1_score(y_test, preds, average="weighted")
        })

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Evaluation completed. Results saved to {output_path}")

    # ===== Generate comparison figure (RQ1) =====
    df_results = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.bar(df_results["Model"], df_results["F1_Score"])
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison (RQ1)")
    plt.tight_layout()

    figure_path = "figures/RQ1_model_comparison.pdf"
    os.makedirs(os.path.dirname(figure_path), exist_ok=True)
    plt.savefig(figure_path)
    plt.close()

    print(f"Figure saved to {figure_path}")


if __name__ == "__main__":
    evaluate_models(FEATURE_DATA_PATH, EVALUATION_OUTPUT_PATH)
