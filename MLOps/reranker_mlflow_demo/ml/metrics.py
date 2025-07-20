from collections import defaultdict
from sklearn.metrics import ndcg_score


def evaluate_model(model, dataset, k=10) -> dict:
    """
    Evaluates a CrossEncoder model on document-query pairs.
    Returns a dictionary of evaluation metrics.
    """
    groups = defaultdict(list)
    for query, doc, label in zip(dataset["query"], dataset["document"], dataset["label"]):
        groups[doc].append((query, label))

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
        ndcg = ndcg_score([labels], [scores], k=k)
        ndcg_list.append(ndcg)

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