import mlflow
import logging

logger = logging.getLogger(__name__)


def log_metrics(metrics: dict):
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value)
            logger.debug(f"Logged metric: {key} = {value}")


def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)
        logger.debug(f"Logged param: {key} = {value}")


def log_artifacts(path: str, artifact_path: str = None):
    mlflow.log_artifacts(path, artifact_path=artifact_path)
    logger.debug(f"Logged artifacts from {path} to {artifact_path or 'root'}")


def log_model_pyfunc(model, output_path: str):
    """
    Logs a CrossEncoder model as an MLflow pyfunc.
    """
    class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder.load(context.artifacts["model_dir"])

        def predict(self, context, model_input):
            return self.model.predict(list(zip(model_input["query"], model_input["document"])))

    mlflow.pyfunc.log_model(
        artifact_path="model_pyfunc",
        python_model=CrossEncoderWrapper(),
        artifacts={"model_dir": output_path},
    )
    logger.info("Logged pyfunc model to 'model_pyfunc'")