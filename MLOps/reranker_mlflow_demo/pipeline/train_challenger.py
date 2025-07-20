import os
import mlflow
import logging

from ml.dataset import load_training_data, prepare_datasets
from ml.model import train_model
from ml.metrics import evaluate_model
from tracking.logger import log_metrics, log_model_pyfunc, log_params, log_artifacts

logger = logging.getLogger(__name__)


def run_challenger_training_pipeline(date_from: str, date_to: str, config: dict) -> dict:
    run_name = f"challenger-{config['base_model'].split('/')[-1]}-{date_from}_to_{date_to}"
    output_path = os.path.join(config['output_dir'], run_name)

    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=run_name) as run:
        log_params({
            "base_model": config["base_model"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "num_negatives": config["num_negatives"],
            "date_from": date_from,
            "date_to": date_to
        })

        train_samples, df = load_training_data(date_from, date_to)
        if not train_samples:
            logger.warning("No training samples found. Exiting.")
            return {}

        log_metrics({
            "num_samples": len(df),
            "positive_labels": df["relevance"].sum(),
            "class_balance": df["relevance"].mean()
        })

        train_dataset, eval_dataset = prepare_datasets(df)

        model = train_model(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_name=config["base_model"],
            output_dir=output_path,
            num_epochs=config["epochs"],
            batch_size=config["batch_size"],
            num_negatives=config["num_negatives"]
        )

        log_artifacts(output_path, artifact_path="model_artifact")
        log_model_pyfunc(model, output_path)

        metrics = evaluate_model(model, eval_dataset)
        log_metrics(metrics)

        model_uri = f"runs:/{run.info.run_id}/model_pyfunc"
        return {
            "model_uri": model_uri,
            "eval_dataset": eval_dataset,
            "metrics": metrics
        }