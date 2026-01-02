import os
import pandas as pd

# Get project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# Input & output paths
INPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_ingested.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "data", "sample", "occupancy_cleaned.csv"
)


def clean_data(input_path: str, output_path: str):
    """
    Cleans ingested building occupancy data and saves cleaned output.
    """

    # Load data
    df = pd.read_csv(input_path)
    print("Initial shape:", df.shape)

    if df.empty:
        raise ValueError("Input data is empty")

    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    print("Shape after cleaning:", df.shape)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)

    print(f"Data cleaning completed. File saved to: {output_path}")


if __name__ == "__main__":
    clean_data(INPUT_PATH, OUTPUT_PATH)
