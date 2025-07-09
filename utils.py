from model_registry import caption_processor, caption_model, text_model, clip_model, clip_processor, rerank_model, DEVICE
from PIL import Image
import requests
import torch
from typing import Optional, List, Dict, Any

from clients.qdrant_client_wrapper import QdrantVectorStoreClient


def load_image(image_source: str) -> Image.Image:
    if image_source.startswith("http://") or image_source.startswith("https://"):
        return Image.open(requests.get(image_source, stream=True).raw).convert("RGB")
    else:
        return Image.open(image_source).convert("RGB")


def generate_image_caption(image_source: str) -> Optional[str]:
    try:
        image = load_image(image_source)
        inputs = caption_processor(images=image, return_tensors="pt").to(DEVICE)
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except:
        return None


def generate_image_embedding(image_source: str) -> Optional[List[float]]:
    try:
        image = load_image(image_source)
        inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        return embedding[0].cpu().tolist()
    except:
        return None


def generate_text_embedding(text: str) -> List[float]:
    return text_model.encode(text, normalize_embeddings=True).tolist()


def generate_combined_embedding_with_metadata(article_text: str,
                                              image_source: Optional[str] = None) -> Dict[str, Optional[Any]]:
    image_caption = None
    image_embedding = None

    if image_source:
        image_caption = generate_image_caption(image_source)
        combined_text = f"[ARTICLE]: {article_text}\n[IMAGE]: {image_caption}"
        image_embedding = generate_image_embedding(image_source)
    else:
        combined_text = f"[ARTICLE]: {article_text}"

    embedding = generate_text_embedding(combined_text)

    return {
        "image_caption": image_caption,
        "combined_text": combined_text,
        "embedding": embedding,
        "image_embedding": image_embedding
    }


def clean_points(points: List) -> List:
    return [p for p in points
            if p.payload.get("combined_text") and len(p.payload["combined_text"]) >= 20 and p.score >= 0.6]


def rerank_results(query: str, candidates: List, top_k: int = 5) -> List:
    pairs = [(query, p.payload.get("combined_text", "")) for p in candidates if p.payload.get("combined_text")]
    scores = rerank_model.predict(pairs)
    scored_points = list(zip(candidates, scores))
    scored_points.sort(key=lambda x: x[1], reverse=True)
    return [point for point, _ in scored_points[:top_k]]


def search_similar_articles(prompt: str, top_k: int = 5, date_from: Optional[str] = None,
                            date_to: Optional[str] = None) -> List:
    qdrant_client = QdrantVectorStoreClient()

    embedding = generate_text_embedding(prompt)

    filters = None
    if date_from and date_to:
        try:
            date_from_int = int(date_from.replace("-", ""))
            date_to_int = int(date_to.replace("-", ""))
            filters = {"date_int": {"gte": date_from_int, "lte": date_to_int}}
        except:
            pass

    candidates = qdrant_client.search_similar_points(query_vector=embedding, top_k=top_k*2, filters=filters, exact=True)
    ranked = rerank_results(query=prompt, candidates=candidates.points, top_k=top_k)

    return clean_points(ranked)


def get_articles_by_date_range(date_from: str, date_to: str) -> List:
    qdrant_client = QdrantVectorStoreClient()

    return qdrant_client.search_points_by_date_range(date_from, date_to)


def find_by_uploaded_image(image_path_or_url: str, prompt: Optional[str] = None) -> List:
    qdrant_client = QdrantVectorStoreClient()

    if prompt:
        text_embedding = generate_text_embedding(prompt)
        sim_threshold = 0.7
    else:
        image_caption = generate_image_caption(image_path_or_url)
        text_embedding = generate_text_embedding(f"[IMAGE]: {image_caption}")
        sim_threshold = 0.65

    candidates = qdrant_client.search_similar_points(query_vector=text_embedding, top_k=10)

    image_embedding = generate_image_embedding(image_path_or_url)
    if not image_embedding:
        return []

    similar = []
    for point in candidates.points:
        stored_vector = point.payload.get("image_embedding")
        if stored_vector:
            sim = torch.nn.functional.cosine_similarity(
                torch.tensor(image_embedding), torch.tensor(stored_vector), dim=0
            ).item()
            if sim > sim_threshold:
                similar.append(point)

    return similar
