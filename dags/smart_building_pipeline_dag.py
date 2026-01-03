from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

# Absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_script(script_path):
    """
    Helper function to run existing pipeline scripts.
    """
    subprocess.run(
        ["python", script_path],
        check=True
    )

# Default DAG arguments
default_args = {
    "owner": "student",
    "start_date": datetime(2024, 1, 1),
    "retries": 0
}

# Define the DAG
with DAG(
    dag_id="smart_building_occupancy_pipeline",
    default_args=default_args,
    schedule=None,          # Manual trigger (Airflow 2.x compliant)
    catchup=False,
    description="End-to-end smart building occupancy prediction pipeline"
) as dag:

    # Task 1: Data Ingestion
    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/data_ingestion/ingest.py")]
    )

    # Task 2: Data Cleaning
    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/data_cleaning/clean.py")]
    )

    # Task 3: Feature Engineering
    transform_features = PythonOperator(
        task_id="transform_features",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/feature_engineering/features.py")]
    )

    # Task 4: Model Training
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/modeling/model.py")]
    )

    # Task 5: Model Evaluation & Output Generation
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/evaluation/evaluate.py")]
    )

    # Define task dependencies (DAG structure)
    extract_data >> clean_data >> transform_features >> train_model >> evaluate_model
