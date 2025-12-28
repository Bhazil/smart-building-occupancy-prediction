from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


def run_ingestion():
    subprocess.run(["python", "src/data_ingestion/ingest.py"], check=True)


def run_cleaning():
    subprocess.run(["python", "src/data_cleaning/clean.py"], check=True)


def run_feature_engineering():
    subprocess.run(["python", "src/feature_engineering/features.py"], check=True)


def run_modeling():
    subprocess.run(["python", "src/modeling/model.py"], check=True)


def run_evaluation():
    subprocess.run(["python", "src/evaluation/evaluate.py"], check=True)


default_args = {
    "owner": "data-engineering-project",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="smart_building_occupancy_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="End-to-end smart building occupancy prediction pipeline",
) as dag:

    ingest_task = PythonOperator(
        task_id="extract_data",
        python_callable=run_ingestion,
    )

    clean_task = PythonOperator(
        task_id="clean_data",
        python_callable=run_cleaning,
    )

    feature_task = PythonOperator(
        task_id="transform_features",
        python_callable=run_feature_engineering,
    )

    model_task = PythonOperator(
        task_id="train_model",
        python_callable=run_modeling,
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=run_evaluation,
    )

    ingest_task >> clean_task >> feature_task >> model_task >> evaluate_task
