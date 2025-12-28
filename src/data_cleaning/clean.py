import os
import pandas as pd

RAW_DATA_PATH = "data/ingested/ingested_data.csv"
CLEAN_DATA_PATH = "data/cleaned/cleaned_data.csv"


def clean_data(input_path: str, output_path: str):
    """
    Cleans raw building sensor data and saves cleaned output.
    """

    # Load data
    df = pd.read_csv(input_path)

    # Basic checks
    if df.empty:
        raise ValueError("Input data is empty")

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    print(f"Data cleaning completed. Clean file saved to: {output_path}")


if __name__ == "__main__":
    clean_data(RAW_DATA_PATH, CLEAN_DATA_PATH)
