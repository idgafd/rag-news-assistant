import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from data_pipeline.batch_parser import run_pipeline_for_range


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_previous_week_range():
    """Returns the date range from Monday to Sunday of the previous week."""
    today = datetime.today()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)
    return last_monday.strftime("%Y-%m-%d"), last_sunday.strftime("%Y-%m-%d")


def run_batch_ingestion(**context):
    """Wrapper function to run the Qdrant pipeline for last week's articles."""
    conf = context.get("dag_run").conf if context.get("dag_run") else {}
    date_from = conf.get("date_from")
    date_to = conf.get("date_to")

    if not date_from or not date_to:
        date_from, date_to = get_previous_week_range()
        logger.info("No date_from/date_to provided, falling back to previous week.")

    logger.info(f"Running Qdrant pipeline from {date_from} to {date_to}")
    run_pipeline_for_range(date_from=date_from, date_to=date_to)
    logger.info("Qdrant pipeline finished successfully.")


with DAG(
    dag_id="qdrant_the_batch_weekly",
    description="Weekly pipeline to ingest The Batch articles into Qdrant",
    start_date=datetime(2024, 1, 1),
    schedule_interval="0 6 * * MON",
    catchup=False,
    max_active_runs=1,
    tags=["qdrant", "the_batch", "weekly"],
) as dag:

    run_pipeline_task = PythonOperator(
        task_id="run_qdrant_batch_pipeline",
        python_callable=run_batch_ingestion,
        provide_context=True,
    )
