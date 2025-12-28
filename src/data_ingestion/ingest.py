"""
Data Ingestion Module
Reads raw occupancy sensor data and saves it in a standardized format.
"""

import os
import pandas as pd

# Paths
RAW_DATA_PATH = "data/sample/occupancy_raw.csv"
OUTPUT_PATH = "data/sample/occupancy_ingested.csv"


def ingest_data(input_path: str, output_path: str):
    """
    Reads raw data, performs basic validation, and saves ingested data.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Raw data file not found at {input_path}")

    df = pd.read_csv(input_path)

    # Basic sanity checks
    if df.empty:
        raise ValueError("Raw data file is empty")

    # Save ingested data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Ingestion completed. File saved to: {output_path}")


if __name__ == "__main__":
    ingest_data(RAW_DATA_PATH, OUTPUT_PATH)
