"""
Smart Building Occupancy Prediction Pipeline
Airflow DAG for Data Engineering – Technical Submission (Part 2)
"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import os
import subprocess

# Project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Helper function to run python scripts
def run_script(script_path):
    subprocess.run(
        ["python", script_path],
        check=True
    )

# Default arguments
default_args = {
    "owner": "data_engineering_student",
    "depends_on_past": False,
    "retries": 0,
}

# DAG definition
with DAG(
    dag_id="smart_building_pipeline",
    description="End-to-end Smart Building Occupancy Prediction Pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule=None,          # modern replacement for schedule_interval
    catchup=False,
    tags=["data-engineering", "ml", "airflow"],
) as dag:

    # Task 1: Data Ingestion
    extract_data = PythonOperator(
        task_id="extract_data",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/data_ingestion/ingest.py")],
    )

    # Task 2: Data Cleaning
    clean_data = PythonOperator(
        task_id="clean_data",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/data_cleaning/clean.py")],
    )

    # Task 3: Feature Engineering
    transform_features = PythonOperator(
        task_id="transform_features",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/feature_engineering/features.py")],
    )

    # Task 4: Model Training
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/modeling/model.py")],
    )

    # Task 5: Model Evaluation
    evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/evaluation/evaluate.py")],
    )

    # Task 6: Generate Figures
    generate_figures = PythonOperator(
        task_id="generate_figures",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/evaluation/generate_figures.py")],
    )

    # Task 7: Generate Tables
    generate_tables = PythonOperator(
        task_id="generate_tables",
        python_callable=run_script,
        op_args=[os.path.join(PROJECT_ROOT, "src/evaluation/generate_tables.py")],
    )

    # DAG Dependencies (Logical Flow)
    (
        extract_data
        >> clean_data
        >> transform_features
        >> train_model
        >> evaluate_model
        >> generate_figures
        >> generate_tables
    )
