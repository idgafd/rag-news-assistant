import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import pandas as pd
from sentence_transformers import InputExample
from datasets import Dataset

from clients.supabase_client_wrapper import SupabaseRerankerClient

logger = logging.getLogger(__name__)


def load_training_data(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_relevance: int = 0,
    max_relevance: int = 1,
) -> Tuple[List[InputExample], pd.DataFrame]:
    """
    Load supervised reranking training data from Supabase based on creation date and relevance bounds.
    """
    if not date_from:
        date_from = (datetime.utcnow() - timedelta(days=30)).date().isoformat()
    if not date_to:
        date_to = datetime.utcnow().date().isoformat()

    logger.info(f"Loading reranking data from Supabase for period: {date_from} to {date_to}")
    logger.info(f"Relevance bounds: {min_relevance} to {max_relevance}")

    client = SupabaseRerankerClient()
    response = client.fetch_examples_by_created_at_range(
        date_from=date_from,
        date_to=date_to,
        min_relevance=min_relevance,
        max_relevance=max_relevance
    )

    rows = response.data
    if not rows:
        logger.warning("No training examples found in the specified date range.")
        return [], pd.DataFrame()

    df = pd.DataFrame(rows)

    train_samples = [
        InputExample(texts=[row["query"], row["document"]], label=float(row["relevance"]))
        for row in rows if row.get("query") and row.get("document")
    ]

    logger.info(f"Loaded {len(train_samples)} training examples.")
    logger.info(f"Label distribution:\n{df['relevance'].value_counts().to_string()}")
    logger.info(f"Unique sources: {df['source'].nunique()} — {df['source'].unique()}")

    return train_samples, df


def prepare_datasets(df, test_size=0.1, seed=42):
    """
    Converts a pandas dataframe to HuggingFace Dataset and splits into train and eval sets.
    """
    dataset = Dataset.from_pandas(
        df[["query", "document", "relevance"]].rename(columns={"relevance": "label"})
    )
    split = dataset.train_test_split(test_size=test_size, seed=seed)
    return split["train"], split["test"]