import logging
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

logger = logging.getLogger(__name__)
client = MlflowClient()


def get_production_model_version(model_name: str) -> ModelVersion:
    """
    Returns the latest production model version.
    Raises an exception if no production model is found.
    """
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(f"No production version found for model '{model_name}'")
    logger.info(f"Found production model version: {versions[0].version}")
    return versions[0]


def promote_model_version(model_name: str, version: int | str) -> None:
    """
    Transitions a model version to Production stage.
    Archives all other versions.
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    logger.info(f"Promoted model '{model_name}' version {version} to Production.")


def archive_model_version(model_name: str, version: int | str) -> None:
    """
    Archives a model version.
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Archived"
    )
    logger.info(f"Archived model '{model_name}' version {version}.")


def get_all_model_versions(model_name: str) -> list[ModelVersion]:
    """
    Returns all versions of the given model.
    """
    versions = client.search_model_versions(f"name='{model_name}'")
    return list(versions)
