import mlflow
import logging
from mlflow.pyfunc import load_model as mlflow_load_model

from ml.metrics import evaluate_model
from tracking.registry import get_production_model_version, promote_model_version, archive_model_version

logger = logging.getLogger(__name__)


def compare_with_production_and_promote(model_uri: str, eval_dataset, model_name: str, metric_key: str = "NDCG@10") -> bool:
    """
    Compares the challenger model with the current production model.
    If the challenger performs better, promote it to production.
    """
    try:
        prod_version_info = get_production_model_version(model_name)
        prod_model = mlflow_load_model(f"models:/{model_name}/{prod_version_info.version}")
        prod_score = evaluate_model(prod_model, eval_dataset).get(metric_key, 0.0)
        logger.info(f"Production model {prod_version_info.version} {metric_key}: {prod_score:.4f}")
    except Exception:
        logger.warning("No production model found. Challenger will be promoted by default.")
        prod_version_info = None
        prod_score = -1.0

    challenger_model = mlflow_load_model(model_uri)
    challenger_score = evaluate_model(challenger_model, eval_dataset).get(metric_key, 0.0)
    logger.info(f"Challenger model {metric_key}: {challenger_score:.4f}")

    if challenger_score > prod_score:
        if prod_version_info:
            archive_model_version(model_name, prod_version_info.version)
        new_version = mlflow.register_model(model_uri, model_name).version
        promote_model_version(model_name, new_version)
        logger.info(f"Promoted challenger to Production as version {new_version}")
        return True

    logger.info("Challenger did not outperform production. No promotion done.")
    return False
