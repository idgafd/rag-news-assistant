import argparse
import pandas as pd
import boto3
import logging

logger = logging.getLogger(__name__)


def load_data_from_s3(s3_path, filename="training_data.csv"):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:]) + f'/{filename}'

    local_path = f"/tmp/{filename}"
    s3.download_file(bucket, key, local_path)
    return pd.read_csv(local_path)


def clean_data(df):
    initial_count = len(df)

    # Remove missing values
    df = df.dropna(subset=['query', 'document', 'relevance'])

    # Remove empty strings
    df = df[df['query'].str.strip() != '']
    df = df[df['document'].str.strip() != '']

    # Validate relevance scores
    df = df[df['relevance'].isin([0, 1])]

    # Remove duplicates
    df = df.drop_duplicates(subset=['query', 'document'])

    # Basic length filtering
    df = df[df['query'].str.len() >= 10]
    df = df[df['document'].str.len() >= 20]
    df = df[df['query'].str.len() <= 500]
    df = df[df['document'].str.len() <= 2000]

    logger.info(f"Data cleaning: {initial_count} -> {len(df)} samples")
    return df


def balance_dataset(df, target_balance=0.3, random_seed=42):
    current_balance = df['relevance'].mean()

    if abs(current_balance - target_balance) < 0.05:
        return df

    positive_samples = df[df['relevance'] == 1]
    negative_samples = df[df['relevance'] == 0]

    if current_balance < target_balance:
        # Undersample negatives
        target_negative = int(len(positive_samples) * (1 - target_balance) / target_balance)
        if target_negative < len(negative_samples):
            negative_samples = negative_samples.sample(n=target_negative, random_state=random_seed)
    else:
        # Undersample positives
        target_positive = int(len(negative_samples) * target_balance / (1 - target_balance))
        if target_positive < len(positive_samples):
            positive_samples = positive_samples.sample(n=target_positive, random_state=random_seed)

    balanced_df = pd.concat([positive_samples, negative_samples], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(f"Balanced dataset: {len(df)} -> {len(balanced_df)} samples")
    return balanced_df


def create_splits(df, test_ratio=0.15, val_ratio=0.15, random_seed=42):
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    positive_samples = df[df['relevance'] == 1]
    negative_samples = df[df['relevance'] == 0]

    n_pos = len(positive_samples)
    n_neg = len(negative_samples)

    test_pos = int(n_pos * test_ratio)
    test_neg = int(n_neg * test_ratio)
    val_pos = int(n_pos * val_ratio)
    val_neg = int(n_neg * val_ratio)

    pos_shuffled = positive_samples.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    neg_shuffled = negative_samples.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Create splits
    test_df = pd.concat([
        pos_shuffled[:test_pos],
        neg_shuffled[:test_neg]
    ], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    val_df = pd.concat([
        pos_shuffled[test_pos:test_pos + val_pos],
        neg_shuffled[test_neg:test_neg + val_neg]
    ], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    train_df = pd.concat([
        pos_shuffled[test_pos + val_pos:],
        neg_shuffled[test_neg + val_neg:]
    ], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_to_s3(df, s3_path, filename):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:]) + f'/{filename}'

    local_path = f"/tmp/{filename}"
    df.to_csv(local_path, index=False)

    s3.upload_file(local_path, bucket, key)
    logger.info(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--eval_output", type=str, required=True)
    parser.add_argument("--test_split_ratio", type=float, default=0.15)
    parser.add_argument("--validation_split_ratio", type=float, default=0.15)
    parser.add_argument("--target_balance", type=float, default=0.3)
    parser.add_argument("--balance_dataset", action="store_true")
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    # Load and clean data
    df = load_data_from_s3(args.input_path)
    df = clean_data(df)

    if df.empty:
        raise ValueError("No valid data after cleaning")

    # Balance if requested
    if args.balance_dataset:
        df = balance_dataset(df, args.target_balance, args.random_seed)

    # Create splits
    train_df, val_df, test_df = create_splits(
        df, args.test_split_ratio, args.validation_split_ratio, args.random_seed
    )

    # Save datasets
    save_to_s3(train_df, args.train_output, "train_data.csv")
    save_to_s3(val_df, args.train_output, "val_data.csv")

    # Combine val and test for evaluation
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    save_to_s3(eval_df, args.eval_output, "eval_data.csv")

    logger.info("Dataset preparation completed")


if __name__ == "__main__":
    main()