import os
import logging
import argparse
from datetime import datetime, timedelta

from datasets import Dataset

from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
    CrossEncoderModelCardData,
)
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

from data_pipeline.load_reranker_data import load_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train reranker model using supervised query-document pairs.")
    parser.add_argument("--base_model", type=str, default="microsoft/MiniLM-L12-H384-uncased", help="Base CrossEncoder model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per device")
    parser.add_argument("--output_dir", type=str, default="models/reranker", help="Base directory to save the trained model")
    parser.add_argument("--date_from", type=str, help="Start date in YYYY-MM-DD format (defaults to 30 days ago)")
    parser.add_argument("--date_to", type=str, help="End date in YYYY-MM-DD format (defaults to today)")
    parser.add_argument("--num_negatives", type=int, default=4, help="Number of negatives for ranking loss")
    return parser.parse_args()


def main():
    args = parse_args()

    date_from = args.date_from or (datetime.utcnow() - timedelta(days=30)).date().isoformat()
    date_to = args.date_to or datetime.utcnow().date().isoformat()
    run_name = f"reranker-{args.base_model.split('/')[-1]}-{date_from}_to_{date_to}"
    output_path = os.path.join(args.output_dir, run_name)

    logger.info(f"Training CrossEncoder from {date_from} to {date_to}")

    # Load dataset from Supabase
    train_samples, df = load_training_data(date_from=date_from, date_to=date_to)
    if not train_samples:
        logger.warning("No training samples found. Exiting.")
        return

    dataset = Dataset.from_pandas(
        df[["query", "document", "relevance"]].rename(columns={"relevance": "label"})
    )
    dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    # Initialize model
    model = CrossEncoder(
        args.base_model,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Fine-tuned CrossEncoder on reranker_dataset",
        ),
    )

    # Loss function
    loss = CachedMultipleNegativesRankingLoss(
        model=model,
        num_negatives=args.num_negatives,
        mini_batch_size=32,
    )

    # (Optional) lightweight evaluator
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "nfcorpus", "nq"],
        batch_size=args.batch_size,
    )
    evaluator(model)

    # Training arguments
    training_args = CrossEncoderTrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        run_name=run_name,
        seed=42,
        fp16=False,
        bf16=True,
    )

    # Trainer
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    trainer.train()
    evaluator(model)

    # Save final model
    model.save_pretrained(output_path)
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
