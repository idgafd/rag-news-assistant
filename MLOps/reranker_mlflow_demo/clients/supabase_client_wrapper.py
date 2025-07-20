from supabase import create_client, Client
from clients.config import SUPABASE_URL, SUPABASE_KEY
from typing import Optional
import datetime
import uuid


class SupabaseRerankerClient:
    def __init__(self):
        self.supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        self.table_name = "reranker_dataset"

    def insert_example(
        self,
        query: str,
        document: str,
        relevance: int,
        source: str,
        document_id: Optional[uuid.UUID] = None,
        document_date: Optional[datetime.date] = None,
    ):
        data = {
            "query": query,
            "document": document,
            "relevance": relevance,
            "source": source,
            "document_id": str(document_id) if document_id else str(uuid.uuid4()),
            "document_date": document_date,
        }
        return self.supabase.table(self.table_name).insert(data).execute()

    def fetch_recent_examples(self, limit: int = 100):
        return (
            self.supabase.table(self.table_name)
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

    def fetch_examples_by_created_at_range(self, date_from: str, date_to: Optional[str] = None, min_relevance: int = 0,
                                           max_relevance: int = 1):
        """
        Fetch examples created between date_from and date_to (inclusive).
        Dates in 'YYYY-MM-DD' format.
        """
        query = (
            self.supabase.table(self.table_name)
            .select("*")
            .gte("created_at", date_from)
            .lte("created_at", date_to or datetime.date.today().isoformat())
            .gte("relevance", min_relevance)
            .lte("relevance", max_relevance)
        )
        return query.execute()

    def get_by_query(self, query: str):
        return (
            self.supabase.table(self.table_name)
            .select("*")
            .eq("query", query)
            .execute()
        )

    def delete_example(self, id: int):
        return (
            self.supabase.table(self.table_name)
            .delete()
            .eq("id", id)
            .execute()
        )

    def delete_by_document_date_range(self, date_from: str, date_to: str):
        """Delete all examples with document_date between date_from and date_to (inclusive)."""
        return (
            self.supabase.table(self.table_name)
            .delete()
            .gte("document_date", date_from)
            .lte("document_date", date_to)
            .execute()
        )
