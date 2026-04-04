# Kept for reference — orchestration handled by pipelines/scheduler.py

"""
Airflow DAG: FailSight end-to-end pipeline
Runs: ingest → features → train → weibull → vectorstore rebuild
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "failsight",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="failsight_pipeline",
    default_args=default_args,
    description="FailSight: ingest → features → train → analyze",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["failsight", "predictive-maintenance"],
) as dag:

    ingest = BashOperator(
        task_id="ingest_raw_data",
        bash_command="cd /opt/failsight && python pipelines/ingest.py",
    )

    features = BashOperator(
        task_id="feature_engineering",
        bash_command="cd /opt/failsight && python pipelines/features.py",
    )

    train = BashOperator(
        task_id="train_rul_model",
        bash_command="cd /opt/failsight && python models/train_rul.py",
    )

    weibull = BashOperator(
        task_id="weibull_analysis",
        bash_command="cd /opt/failsight && python analysis/weibull_analysis.py",
    )

    vectorstore = BashOperator(
        task_id="rebuild_vectorstore",
        bash_command="cd /opt/failsight && python agent/build_vectorstore.py",
    )

    ingest >> features >> train >> [weibull, vectorstore]