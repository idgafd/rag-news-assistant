from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, CrossEncoderModelCardData
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss
import os


def train_model(
    train_dataset,
    eval_dataset,
    model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    num_negatives: int
):
    model = CrossEncoder(
        model_name,
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
        learning_rate=2e-5,
        warmup_ratio=0.1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=50,
        logging_first_step=True,
        run_name=os.path.basename(output_dir),
        seed=42,
        fp16=False,
        bf16=False,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss_fn,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model