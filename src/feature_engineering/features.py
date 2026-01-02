import os
import pandas as pd

# --------------------------------------------------
# Get project root directory
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

# --------------------------------------------------
# Input & output paths
# --------------------------------------------------
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_features.csv"
)


def engineer_features(input_path: str, output_path: str):
    """
    Creates engineered features from cleaned occupancy data
    and saves feature dataset.
    """

    # -----------------------------
    # Load cleaned data
    # -----------------------------
    df = pd.read_csv(input_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    print("Columns:", df.columns.tolist())
    print("Initial shape:", df.shape)

    if df.empty:
        raise ValueError("Cleaned data is empty")

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # CO2 rolling mean
    df["co2_rolling_mean"] = (
        df["co2"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    # Temperature & humidity interaction
    df["temp_humidity_interaction"] = (
        df["temperature"] * df["humidity"]
    )

    # Light level indicator
    df["light_on"] = (df["light"] > 0).astype(int)

    # -----------------------------
    # Final checks
    # -----------------------------
    print("Shape after feature engineering:", df.shape)

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save feature dataset
    df.to_csv(output_path, index=False)

    print(f"Feature engineering completed. File saved to: {output_path}")


if __name__ == "__main__":
    engineer_features(INPUT_PATH, OUTPUT_PATH)
