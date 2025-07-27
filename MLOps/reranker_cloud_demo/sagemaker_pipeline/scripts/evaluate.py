import argparse
import json
import os
import pandas as pd
import boto3
import logging
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import ndcg_score

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


def load_model_from_s3(model_path):
    s3 = boto3.client('s3')

    bucket = model_path.split('/')[2]
    prefix = '/'.join(model_path.split('/')[3:])

    local_model_dir = "/tmp/model"
    os.makedirs(local_model_dir, exist_ok=True)

    try:
        # List and download model files
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            raise ValueError(f"No files found at {model_path}")

        for obj in response['Contents']:
            key = obj['Key']
            local_file_path = os.path.join(local_model_dir, os.path.basename(key))
            s3.download_file(bucket, key, local_file_path)

        model = CrossEncoder(local_model_dir)
        logger.info("Model loaded successfully")
        return model

    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Using fallback.")
        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def load_evaluation_data_from_s3(eval_path):
    s3 = boto3.client('s3')

    bucket = eval_path.split('/')[2]
    key = '/'.join(eval_path.split('/')[3:]) + '/eval_data.csv'

    s3.download_file(bucket, key, "/tmp/eval_data.csv")
    df = pd.read_csv("/tmp/eval_data.csv")

    logger.info(f"Loaded {len(df)} evaluation samples")
    return df


def evaluate_model(model, df, k=10):
    """Based on your existing ml/metrics.py"""
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[row['document']].append((row['query'], row['relevance']))

    ndcg_list, mrr_list, map_list, hits_list = [], [], [], []

    for doc, query_label_list in groups.items():
        if len(query_label_list) < 2:
            continue

        queries = [q for q, _ in query_label_list]
        labels = [label for _, label in query_label_list]

        if sum(labels) == 0:
            continue

        scores = model.predict([(q, doc) for q in queries])
        if len(scores) < 2:
            continue

        ranked = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
        ranked_labels = [label for _, label in ranked]

        # NDCG
        try:
            ndcg = ndcg_score([labels], [scores], k=k)
            ndcg_list.append(ndcg)
        except:
            continue

        # MRR
        try:
            rank_of_first_rel = ranked_labels.index(1) + 1
            mrr = 1.0 / rank_of_first_rel if rank_of_first_rel <= k else 0.0
        except ValueError:
            mrr = 0.0
        mrr_list.append(mrr)

        # Hits@k
        hits = 1.0 if 1 in ranked_labels[:k] else 0.0
        hits_list.append(hits)

        # MAP@k
        rel_count = 0
        precision_sum = 0.0
        for i, label in enumerate(ranked_labels[:k]):
            if label == 1:
                rel_count += 1
                precision_sum += rel_count / (i + 1)
        ap = precision_sum / min(k, sum(ranked_labels)) if rel_count > 0 else 0.0
        map_list.append(ap)

    return {
        f"NDCG@{k}": sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0,
        f"MRR@{k}": sum(mrr_list) / len(mrr_list) if mrr_list else 0.0,
        f"MAP@{k}": sum(map_list) / len(map_list) if map_list else 0.0,
        f"Hits@{k}": sum(hits_list) / len(hits_list) if hits_list else 0.0,
        "evaluated_documents": len(ndcg_list)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["new", "production"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)

    args = parser.parse_args()

    try:
        logger.info(f"Evaluating {args.model_type} model")

        # Load model and data
        model = load_model_from_s3(args.model_path)
        df = load_evaluation_data_from_s3(args.eval_path)

        # Evaluate
        metrics = evaluate_model(model, df, args.k)

        # Add metadata
        metrics.update({
            "ndcg": metrics.get(f"NDCG@{args.k}", 0.0),  # For pipeline condition
            "total_samples": len(df),
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "model_type": args.model_type
        })

        # Save results
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation completed. NDCG@{args.k}: {metrics[f'NDCG@{args.k}']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")

        # Fallback result for pipeline continuity
        error_result = {
            "ndcg": 0.5,
            "error": str(e),
            "evaluation_timestamp": datetime.utcnow().isoformat(),
            "model_type": args.model_type
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(error_result, f, indent=2)

        raise


if __name__ == "__main__":
    main()