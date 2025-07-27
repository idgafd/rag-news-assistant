import argparse
import boto3
import mlflow
import logging
from datetime import datetime
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def download_model_from_s3(s3_path):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:])

    local_model_dir = "/tmp/deployment_model"
    import os
    os.makedirs(local_model_dir, exist_ok=True)

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

    if 'Contents' not in response:
        raise ValueError(f"No files found at {s3_path}")

    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('/') or key.endswith('.json'):
            continue

        local_file_path = os.path.join(local_model_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_file_path)

    logger.info(f"Model downloaded to: {local_model_dir}")
    return local_model_dir


def register_model_in_mlflow(model_path, model_name):
    model_dir = download_model_from_s3(model_path)

    run_name = f"model_registration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_source", model_path)
        mlflow.log_param("registration_timestamp", datetime.utcnow().isoformat())

        # Log model using same wrapper as your existing code
        class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(context.artifacts["model_path"])

            def predict(self, context, model_input):
                if hasattr(model_input, 'values'):
                    if len(model_input.columns) >= 2:
                        pairs = list(zip(model_input.iloc[:, 0], model_input.iloc[:, 1]))
                    else:
                        raise ValueError("Input DataFrame must have at least 2 columns")
                elif isinstance(model_input, list):
                    pairs = model_input
                else:
                    raise ValueError("Input must be DataFrame or list of [query, document] pairs")

                return self.model.predict(pairs)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CrossEncoderWrapper(),
            artifacts={"model_path": model_dir},
            input_example=[["What is machine learning?", "Machine learning is a subset of AI"]],
        )

        return f"runs:/{run.info.run_id}/model"


def promote_model_to_production(model_uri, model_name):
    client = MlflowClient()

    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "deployment_timestamp": datetime.utcnow().isoformat(),
            "source": "sagemaker_pipeline"
        }
    )

    # Archive current production model if exists
    try:
        current_prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for version in current_prod_versions:
            logger.info(f"Archiving current production model version {version.version}")
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
    except Exception as e:
        logger.warning(f"No current production model to archive: {e}")

    # Promote new model
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )

    logger.info(f"Model version {model_version.version} promoted to Production")
    return model_version.version


def copy_model_to_production_s3(source_s3_path, production_s3_path):
    if not production_s3_path:
        return

    s3 = boto3.client('s3')

    source_bucket = source_s3_path.split('/')[2]
    source_prefix = '/'.join(source_s3_path.split('/')[3:])

    dest_bucket = production_s3_path.split('/')[2]
    dest_prefix = '/'.join(production_s3_path.split('/')[3:])

    response = s3.list_objects_v2(Bucket=source_bucket, Prefix=source_prefix)

    if 'Contents' not in response:
        raise ValueError(f"No files found at source: {source_s3_path}")

    copied_files = 0
    for obj in response['Contents']:
        source_key = obj['Key']
        relative_path = source_key[len(source_prefix):].lstrip('/')
        dest_key = f"{dest_prefix}/{relative_path}" if dest_prefix else relative_path

        copy_source = {'Bucket': source_bucket, 'Key': source_key}
        s3.copy_object(CopySource=copy_source, Bucket=dest_bucket, Key=dest_key)
        copied_files += 1

    logger.info(f"Copied {copied_files} files to production location")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--production_s3_path", type=str, default=None)
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="CrossEncoderReranker")

    args = parser.parse_args()

    # Set up MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    logger.info(f"Deploying model from {args.model_path}")

    try:
        # Register model
        model_uri = register_model_in_mlflow(args.model_path, args.model_name)

        # Promote to production
        model_version = promote_model_to_production(model_uri, args.model_name)

        # Copy to production S3 if specified
        if args.production_s3_path:
            copy_model_to_production_s3(args.model_path, args.production_s3_path)

        logger.info("Model deployment completed successfully")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()