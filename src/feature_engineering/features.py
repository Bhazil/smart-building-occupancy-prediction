import os
import pandas as pd

# Paths
CLEAN_DATA_PATH = "data/processed/clean_data.csv"
FEATURE_DATA_PATH = "data/processed/feature_data.csv"


def engineer_features(input_path: str, output_path: str):
    """
    Perform basic time-series feature engineering
    """
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Clean data file is empty")

    # Convert timestamp if exists
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.day
        df["weekday"] = df["timestamp"].dt.weekday

    # Rolling averages (example for sensors)
    sensor_cols = [col for col in df.columns if col not in ["timestamp", "occupancy"]]

    for col in sensor_cols:
        df[f"{col}_rolling_mean"] = df[col].rolling(window=3).mean()

    df = df.dropna()

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save feature-engineered data
    df.to_csv(output_path, index=False)

    print(f"Feature engineering completed. File saved to: {output_path}")


if __name__ == "__main__":
    engineer_features(CLEAN_DATA_PATH, FEATURE_DATA_PATH)
