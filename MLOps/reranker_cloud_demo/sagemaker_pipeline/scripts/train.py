import argparse
import os
import pandas as pd
import boto3
import mlflow
import logging
from datetime import datetime
from datasets import Dataset

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, \
    CrossEncoderModelCardData
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss

logger = logging.getLogger(__name__)


def load_training_data_from_s3(s3_path):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    prefix = '/'.join(s3_path.split('/')[3:])

    # Download files
    s3.download_file(bucket, f"{prefix}/train_data.csv", "/tmp/train_data.csv")
    s3.download_file(bucket, f"{prefix}/val_data.csv", "/tmp/val_data.csv")

    train_df = pd.read_csv("/tmp/train_data.csv")
    val_df = pd.read_csv("/tmp/val_data.csv")

    logger.info(f"Loaded {len(train_df)} train, {len(val_df)} val samples")

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(
        train_df[["query", "document", "relevance"]].rename(columns={"relevance": "label"})
    )
    val_dataset = Dataset.from_pandas(
        val_df[["query", "document", "relevance"]].rename(columns={"relevance": "label"})
    )

    return train_dataset, val_dataset


def train_crossencoder_model(train_dataset, val_dataset, base_model, output_dir,
                             num_epochs, batch_size, learning_rate, num_negatives):
    os.makedirs(output_dir, exist_ok=True)

    model = CrossEncoder(
        base_model,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Fine-tuned CrossEncoder on reranker dataset",
        ),
    )

    loss_fn = CachedMultipleNegativesRankingLoss(
        model=model,
        num_negatives=num_negatives,
        mini_batch_size=32,
    )

    training_args = CrossEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        run_name=f"crossencoder_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        seed=42,
        fp16=False,
        bf16=False,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss_fn,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model


def save_model_to_s3(output_dir, s3_output_path):
    s3 = boto3.client('s3')

    bucket = s3_output_path.split('/')[2]
    prefix = '/'.join(s3_output_path.split('/')[3:])

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, output_dir)
            s3_key = f"{prefix}/{relative_path}" if prefix else relative_path

            s3.upload_file(local_path, bucket, s3_key)

    logger.info("Model saved to S3")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--output_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_negatives", type=int, default=4)
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default="SageMaker_CrossEncoder_Training")

    args = parser.parse_args()

    # Set up MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(args.experiment_name)

    run_name = f"crossencoder_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("base_model", args.base_model)
        mlflow.log_param("num_epochs", args.num_epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.learning_rate)
        mlflow.log_param("num_negatives", args.num_negatives)

        # Load data
        train_dataset, val_dataset = load_training_data_from_s3(args.train_path)

        # Log dataset stats
        mlflow.log_metric("train_samples", len(train_dataset))
        mlflow.log_metric("val_samples", len(val_dataset))

        train_labels = [item["label"] for item in train_dataset]
        val_labels = [item["label"] for item in val_dataset]

        mlflow.log_metric("train_positive_ratio", sum(train_labels) / len(train_labels))
        mlflow.log_metric("val_positive_ratio", sum(val_labels) / len(val_labels))

        # Train model
        local_output_dir = "/tmp/crossencoder_model"
        model = train_crossencoder_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            base_model=args.base_model,
            output_dir=local_output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_negatives=args.num_negatives
        )

        # Log model
        class CrossEncoderWrapper(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(context.artifacts["model_path"])

            def predict(self, context, model_input):
                if hasattr(model_input, 'values'):
                    pairs = list(zip(model_input.iloc[:, 0], model_input.iloc[:, 1]))
                else:
                    pairs = model_input
                return self.model.predict(pairs)

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=CrossEncoderWrapper(),
            artifacts={"model_path": local_output_dir},
            input_example=[["What is machine learning?", "Machine learning is a subset of AI"]],
        )

        # Save to S3
        save_model_to_s3(local_output_dir, args.output_model)

        logger.info("Training completed successfully")


if __name__ == "__main__":
    main()