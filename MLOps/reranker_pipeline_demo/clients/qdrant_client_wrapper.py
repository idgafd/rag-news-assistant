from typing import List, Optional, Dict, Any, Union
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse, Record
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, Range, FilterSelector,
    SearchParams, MatchValue
)

from clients.config import QDRANT_API_KEY, QDRANT_URL, QDRANT_COLLECTION


class QdrantVectorStoreClient:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        self.collection_name = QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int):
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                print(f"Created collection '{self.collection_name}'")

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="date_int",
                field_schema="integer"
            )
            print("Index on 'date_int' ensured.")
        except Exception as e:
            print(f"Failed to ensure collection: {e}")

    def upload_points(self, points: List[Dict[str, Any]]):
        try:
            if not points:
                print("No points to upload.")
                return

            self.ensure_collection(vector_size=len(points[0]["vector"]))

            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                    for p in points
                ]
            )
            print(f"Uploaded {len(points)} points to '{self.collection_name}'")
        except Exception as e:
            print(f"Failed to upload points: {e}")

    def delete_all_points(self, filter: Optional[dict] = None):
        try:
            if self.client.collection_exists(self.collection_name):
                selector = FilterSelector(
                    filter=Filter(**filter) if filter else Filter()
                )
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=selector
                )
                print(f"Deleted points in '{self.collection_name}' (filtered: {bool(filter)})")
            else:
                print("Collection does not exist.")
        except Exception as e:
            print(f"Failed to delete points: {e}")

    def get_all_points(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                print("Collection does not exist.")
                return []

            return self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=None,
                limit=2000,
                with_vectors=True
            )[0]
        except Exception as e:
            print(f"Failed to retrieve all points: {e}")
            return []

    def search_similar_points(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        hnsw_ef: int = 128,
        exact: bool = False,
    ) -> Union[List[Any], QueryResponse]:
        try:
            if not self.client.collection_exists(self.collection_name):
                print("Collection does not exist.")
                return []

            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, dict) and set(value.keys()) == {"gte", "lte"}:
                        conditions.append(
                            FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"]))
                        )
                    else:
                        conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )
                qdrant_filter = Filter(must=conditions)

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=hnsw_ef, exact=exact),
                query_filter=qdrant_filter
            )
            return results
        except Exception as e:
            print(f"Failed to search similar points: {e}")
            return []

    def search_points_by_date_range(self, date_from: str, date_to: str):
        try:
            if not self.client.collection_exists(self.collection_name):
                print("Collection does not exist.")
                return []

            date_from_int = int(date_from.replace("-", ""))
            date_to_int = int(date_to.replace("-", ""))

            return self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="date_int",
                            range=Range(gte=date_from_int, lte=date_to_int)
                        )
                    ]
                ),
                limit=2000
            )[0]
        except Exception as e:
            print(f"Failed to search by date range: {e}")
            return []

    def get_latest_date_int(self) -> Optional[int]:
        try:
            if not self.client.collection_exists(self.collection_name):
                print("Collection does not exist.")
                return None

            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=2000,
                with_payload=True,
                scroll_filter=None,
            )

            date_ints = [
                point.payload.get("date_int") for point in results
                if point.payload and "date_int" in point.payload
            ]

            return max(date_ints) if date_ints else None
        except Exception as e:
            print(f"Failed to get latest date_int: {e}")
            return None

    def get_points_for_latest_date(self) -> Union[list[Any], list[Record]]:
        try:
            latest_date = self.get_latest_date_int()
            if latest_date is None:
                print("No data available to determine latest date.")
                return []

            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="date_int",
                            match=MatchValue(value=latest_date)
                        )
                    ]
                ),
                limit=2000,
                with_vectors=True
            )
            return results
        except Exception as e:
            print(f"Failed to get points for latest date: {e}")
            return []
