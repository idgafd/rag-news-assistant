from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

print("PYTHONPATH:", sys.path)
print("FILES in /opt/airflow:", os.listdir("/opt/airflow"))

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'reranker_data_pipeline',
    default_args=default_args,
    description='Pipeline for reranker data collection and training',
    schedule_interval=timedelta(days=1),  # Daily execution
    catchup=False,
    max_active_runs=1,
)


def scrape_batch_data(**context):
    """
    Scrape new articles from deeplearning.ai batch
    """
    from datetime import datetime, timedelta
    import os, sys
    print("PYTHONPATH:", sys.path)
    print("FILES in /opt/airflow:", os.listdir("/opt/airflow"))
    with open('requirements.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            print(line.strip())

    # Calculate date range for last 2 days
    execution_date = context['execution_date']
    date_to = execution_date.strftime('%Y-%m-%d')
    date_from = (execution_date - timedelta(days=2)).strftime('%Y-%m-%d')

    # Import scraping function
    from parse_raw_batch_data import run_pipeline_for_range

    print(f"Scraping data from {date_from} to {date_to}")
    run_pipeline_for_range(date_from, date_to, max_pages=5)

    return f"Scraped data for {date_from} to {date_to}"


def generate_training_data(**context):
    """
    Generate synthetic training data from scraped articles
    """
    from datetime import datetime, timedelta

    # Calculate date range
    execution_date = context['execution_date']
    date_to = execution_date.strftime('%Y-%m-%d')
    date_from = (execution_date - timedelta(days=2)).strftime('%Y-%m-%d')

    # Import data generation function
    from generate_training_data import run_generation_pipeline

    print(f"Generating training data from {date_from} to {date_to}")
    run_generation_pipeline(date_from, date_to, clean_before_insert=False)

    return f"Generated training data for {date_from} to {date_to}"


# def check_training_data(**context):
#     """
#     Check if we have enough training data to trigger model training
#     """
#     # You can implement logic to check data volume
#     # For now, we'll trigger training every 7 days
#     execution_date = context['execution_date']
#     day_of_week = execution_date.weekday()
#
#     if day_of_week == 6:  # Sunday
#         return "trigger_training"
#     else:
#         return "skip_training"
#
#
# def train_model(**context):
#     """
#     Train the reranker model
#     """
#     from datetime import datetime, timedelta
#
#     # Calculate date range for training (last 30 days)
#     execution_date = context['execution_date']
#     date_to = execution_date.strftime('%Y-%m-%d')
#     date_from = (execution_date - timedelta(days=30)).strftime('%Y-%m-%d')
#
#     # Import your training function
#     from train_reranker import main
#     import argparse
#
#     # Create args object
#     args = argparse.Namespace(
#         base_model="microsoft/MiniLM-L12-H384-uncased",
#         epochs=2,
#         batch_size=32,
#         output_dir="/opt/airflow/models/reranker",
#         date_from=date_from,
#         date_to=date_to,
#         num_negatives=4
#     )
#
#     print(f"Training model with data from {date_from} to {date_to}")
#     # You'll need to modify your main function to accept args directly
#     # or create a wrapper function
#
#     return f"Trained model with data from {date_from} to {date_to}"


# Define tasks
scrape_task = PythonOperator(
    task_id='scrape_batch_data',
    python_callable=scrape_batch_data,
    dag=dag,
)

generate_data_task = PythonOperator(
    task_id='generate_training_data',
    python_callable=generate_training_data,
    dag=dag,
)

# check_data_task = PythonOperator(
#     task_id='check_training_data',
#     python_callable=check_training_data,
#     dag=dag,
# )
#
# train_task = PythonOperator(
#     task_id='train_model',
#     python_callable=train_model,
#     dag=dag,
# )

# Define task dependencies
# scrape_task >> generate_data_task >> check_data_task >> train_task
scrape_task >> generate_data_task