import argparse
import pandas as pd
import boto3
from datetime import datetime, timedelta
import logging

from supabase_client_wrapper import SupabaseRerankerClient

logger = logging.getLogger(__name__)


def load_training_data(date_from=None, date_to=None, min_relevance=0, max_relevance=1):
    if not date_from:
        date_from = (datetime.utcnow() - timedelta(days=30)).date().isoformat()
    if not date_to:
        date_to = datetime.utcnow().date().isoformat()

    logger.info(f"Loading data from {date_from} to {date_to}")

    client = SupabaseRerankerClient()
    response = client.fetch_examples_by_created_at_range(
        date_from=date_from,
        date_to=date_to,
        min_relevance=min_relevance,
        max_relevance=max_relevance
    )

    if not response.data:
        logger.warning("No data found")
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    logger.info(f"Loaded {len(df)} samples")

    return df


def load_shift_data(lookback_days=7):
    date_to = datetime.utcnow().date().isoformat()
    date_from = (datetime.utcnow() - timedelta(days=lookback_days)).date().isoformat()

    df = load_training_data(date_from=date_from, date_to=date_to)

    if df.empty:
        return df

    # Add features for drift detection
    df['query_length'] = df['query'].str.len()
    df['document_length'] = df['document'].str.len()
    df['query_word_count'] = df['query'].str.split().str.len()
    df['document_word_count'] = df['document'].str.split().str.len()

    return df


def save_to_s3(df, s3_path, filename):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:]) + f'/{filename}'

    local_path = f"/tmp/{filename}"
    df.to_csv(local_path, index=False)

    s3.upload_file(local_path, bucket, key)
    logger.info(f"Saved to s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "shift"])
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--date_from", type=str, default=None)
    parser.add_argument("--date_to", type=str, default=None)
    parser.add_argument("--lookback_days", type=int, default=7)

    args = parser.parse_args()

    if args.mode == "train":
        df = load_training_data(args.date_from, args.date_to)
        if not df.empty:
            save_to_s3(df, args.output_path, "training_data.csv")

    elif args.mode == "shift":
        df = load_shift_data(args.lookback_days)
        if not df.empty:
            save_to_s3(df, args.output_path, "shift_data.csv")


if __name__ == "__main__":
    main()