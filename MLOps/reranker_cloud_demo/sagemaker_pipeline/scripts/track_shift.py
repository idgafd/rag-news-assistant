import argparse
import json
import pandas as pd
import numpy as np
import boto3
from datetime import datetime, timedelta
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


def load_data_from_s3(s3_path, filename="shift_data.csv"):
    s3 = boto3.client('s3')

    bucket = s3_path.split('/')[2]
    key = '/'.join(s3_path.split('/')[3:]) + f'/{filename}'

    local_path = f"/tmp/{filename}"
    try:
        s3.download_file(bucket, key, local_path)
        return pd.read_csv(local_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def calculate_psi(baseline, current, bins=10):
    try:
        _, bin_edges = np.histogram(baseline, bins=bins)
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        baseline_probs = baseline_counts / baseline_counts.sum()
        current_probs = current_counts / current_counts.sum()

        epsilon = 1e-6
        baseline_probs = np.maximum(baseline_probs, epsilon)
        current_probs = np.maximum(current_probs, epsilon)

        psi = ((current_probs - baseline_probs) * np.log(current_probs / baseline_probs)).sum()
        return psi
    except:
        return 0.0


def detect_statistical_drift(baseline_data, current_data):
    drift_results = {}
    features = ['query_length', 'document_length', 'query_word_count', 'document_word_count']

    for feature in features:
        if feature in baseline_data.columns and feature in current_data.columns:
            ks_stat, ks_p = stats.ks_2samp(
                baseline_data[feature].dropna(),
                current_data[feature].dropna()
            )

            psi_score = calculate_psi(
                baseline_data[feature].dropna(),
                current_data[feature].dropna()
            )

            drift_results[feature] = {
                'ks_statistic': float(ks_stat),
                'ks_p_value': float(ks_p),
                'psi_score': float(psi_score),
                'drift_detected': ks_p < 0.05 or psi_score > 0.2
            }

    return drift_results


def detect_text_drift(baseline_data, current_data):
    drift_results = {}

    for text_col in ['query', 'document']:
        if text_col in baseline_data.columns:
            try:
                baseline_texts = baseline_data[text_col].dropna().tolist()
                current_texts = current_data[text_col].dropna().tolist()

                if len(baseline_texts) > 500:
                    baseline_texts = np.random.choice(baseline_texts, 500, replace=False).tolist()
                if len(current_texts) > 500:
                    current_texts = np.random.choice(current_texts, 500, replace=False).tolist()

                all_texts = baseline_texts + current_texts

                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_texts)

                baseline_vectors = tfidf_matrix[:len(baseline_texts)]
                current_vectors = tfidf_matrix[len(baseline_texts):]

                baseline_centroid = baseline_vectors.mean(axis=0)
                current_centroid = current_vectors.mean(axis=0)

                similarity = cosine_similarity(baseline_centroid, current_centroid)[0][0]
                drift_score = 1 - similarity

                drift_results[text_col] = {
                    'cosine_similarity': float(similarity),
                    'drift_score': float(drift_score),
                    'drift_detected': drift_score > 0.3
                }
            except Exception as e:
                logger.warning(f"Text drift calculation failed for {text_col}: {e}")
                drift_results[text_col] = {'drift_detected': False}

    return drift_results


def detect_label_drift(baseline_data, current_data):
    if 'relevance' not in baseline_data.columns:
        return {}

    baseline_dist = baseline_data['relevance'].value_counts(normalize=True).sort_index()
    current_dist = current_data['relevance'].value_counts(normalize=True).sort_index()

    all_labels = sorted(set(baseline_dist.index) | set(current_dist.index))
    baseline_probs = [baseline_dist.get(label, 0) for label in all_labels]
    current_probs = [current_dist.get(label, 0) for label in all_labels]

    try:
        chi2_stat, chi2_p = stats.chisquare(current_probs, baseline_probs)
        kl_div = stats.entropy(current_probs, baseline_probs)
    except:
        chi2_stat, chi2_p, kl_div = 0.0, 1.0, 0.0

    return {
        'chi2_statistic': float(chi2_stat),
        'chi2_p_value': float(chi2_p),
        'kl_divergence': float(kl_div),
        'drift_detected': chi2_p < 0.05 or kl_div > 0.1
    }


def run_drift_detection(data_path, comparison_window_days=7, threshold=0.25):
    df = load_data_from_s3(data_path)

    if df.empty:
        return {
            "shift_detected": False,
            "shift_score": 0.0,
            "error": "No data available"
        }

    df['created_at'] = pd.to_datetime(df['created_at'])

    cutoff_date = df['created_at'].max() - timedelta(days=comparison_window_days)
    baseline_data = df[df['created_at'] <= cutoff_date]
    current_data = df[df['created_at'] > cutoff_date]

    if len(baseline_data) < 10 or len(current_data) < 10:
        return {
            "shift_detected": True,
            "shift_score": 0.3,
            "warning": "Insufficient data"
        }

    statistical_drift = detect_statistical_drift(baseline_data, current_data)
    text_drift = detect_text_drift(baseline_data, current_data)
    label_drift = detect_label_drift(baseline_data, current_data)

    drift_scores = []

    for results in statistical_drift.values():
        if results['drift_detected']:
            drift_scores.append(min(results['psi_score'], 1.0))

    for results in text_drift.values():
        if results.get('drift_detected', False):
            drift_scores.append(results.get('drift_score', 0))

    if label_drift.get('drift_detected', False):
        drift_scores.append(min(label_drift['kl_divergence'], 1.0))

    overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
    drift_detected = overall_drift_score >= threshold

    return {
        "shift_detected": drift_detected,
        "shift_score": float(overall_drift_score),
        "statistical_drift": statistical_drift,
        "text_drift": text_drift,
        "label_drift": label_drift,
        "detection_timestamp": datetime.utcnow().isoformat()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--comparison_window_days", type=int, default=7)
    parser.add_argument("--drift_threshold", type=float, default=0.25)

    args = parser.parse_args()

    results = run_drift_detection(
        args.input_path,
        args.comparison_window_days,
        args.drift_threshold
    )

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    if results["shift_detected"]:
        logger.warning(f"Data drift detected! Score: {results['shift_score']:.4f}")
    else:
        logger.info(f"No drift detected. Score: {results['shift_score']:.4f}")


if __name__ == "__main__":
    main()