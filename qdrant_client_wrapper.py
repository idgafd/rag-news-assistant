from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse
from qdrant_client.models import (Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, FilterSelector,
                                  SearchParams, MatchValue)
from config import QDRANT_API_KEY, QDRANT_URL, QDRANT_COLLECTION


client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def ensure_collection(vector_size: int):
    try:
        if not client.collection_exists(QDRANT_COLLECTION):
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(f"Created collection '{QDRANT_COLLECTION}'")

        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="date_int",
            field_schema="integer"
        )
        print("Index on 'date_int' ensured.")
    except Exception as e:
        print(f"Failed to ensure collection: {e}")


def upload_points(points: list):
    try:
        if not points:
            print("No points to upload.")
            return

        ensure_collection(vector_size=len(points[0]["vector"]))

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points
            ]
        )
        print(f"Uploaded {len(points)} points to '{QDRANT_COLLECTION}'")
    except Exception as e:
        print(f"Failed to upload points: {e}")


def delete_all_points(filter: dict = None):
    try:
        if client.collection_exists(QDRANT_COLLECTION):
            if filter:
                parsed_filter = Filter(**filter)
                selector = FilterSelector(filter=parsed_filter)
            else:
                selector = FilterSelector(filter=Filter())

            client.delete(collection_name=QDRANT_COLLECTION, points_selector=selector)
            print(f"Deleted points in '{QDRANT_COLLECTION}' (filtered: {bool(filter)})")
        else:
            print("Collection does not exist.")
    except Exception as e:
        print(f"Failed to delete points: {e}")


def get_all_points():
    try:
        if not client.collection_exists(QDRANT_COLLECTION):
            print("Collection does not exist.")
            return []

        return client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=None,
            limit=2000,
            with_vectors=True
        )[0]
    except Exception as e:
        print(f"Failed to retrieve all points: {e}")
        return []


def search_similar_points(
    query_vector: List[float],
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    hnsw_ef: int = 128,
    exact: bool = False,
) -> list[Any] | QueryResponse:
    try:
        if not client.collection_exists(QDRANT_COLLECTION):
            print("Collection does not exist.")
            return []

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if isinstance(value, dict) and set(value.keys()) == {"gte", "lte"}:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=Range(gte=value["gte"], lte=value["lte"])
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            qdrant_filter = Filter(must=conditions)

        results = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=hnsw_ef, exact=exact),
            query_filter=qdrant_filter
        )

        return results

    except Exception as e:
        print(f"Failed to search similar points: {e}")
        return []


def search_points_by_date_range(date_from: str, date_to: str):
    try:
        if not client.collection_exists(QDRANT_COLLECTION):
            print("Collection does not exist.")
            return []

        date_from_int = int(date_from.replace("-", ""))
        date_to_int = int(date_to.replace("-", ""))

        return client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="date_int",
                        range=Range(
                            gte=date_from_int,
                            lte=date_to_int
                        )
                    )
                ]
            ),
            limit=2000
        )[0]
    except Exception as e:
        print(f"Failed to search by date range: {e}")
        return []
