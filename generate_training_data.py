import uuid
import re
import json
import logging
from typing import List, Dict

from clients.qdrant_client_wrapper import QdrantVectorStoreClient
from clients.supabase_client_wrapper import SupabaseRerankerClient
from clients.openai_client_wrapper import OpenAIRouterClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_examples(document_text: str, document_id: uuid.UUID, document_date: str,
                     queries_dict: Dict[str, List[str]]) -> List[Dict]:
    """Transforms a dictionary of relevant and non-relevant queries into structured
    training examples for insertion into the reranker dataset"""
    rows = []
    for q in queries_dict.get("relevant", []):
        rows.append({
            "query": q,
            "document": document_text,
            "document_id": document_id,
            "document_date": document_date,
            "relevance": 1,
            "source": "generated"
        })
    for q in queries_dict.get("non_relevant", []):
        rows.append({
            "query": q,
            "document": document_text,
            "document_id": document_id,
            "document_date": document_date,
            "relevance": 0,
            "source": "generated"
        })
    return rows


def parse_model_json_response(raw_response: str) -> Dict[str, List[str]]:
    """Parses a JSON block from the LLM response, removing markdown syntax if present"""
    if not raw_response or not raw_response.strip():
        raise ValueError("Empty response from model.")

    cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", raw_response.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON. Raw cleaned string:\n%s", cleaned)
        raise e


def run_generation_pipeline(date_from: str, date_to: str, clean_before_insert: bool = True) -> None:
    """Main pipeline that retrieves documents from Qdrant, uses GPT to generate queries,
    and stores (query, document, relevance) examples in Supabase"""
    qdrant_client = QdrantVectorStoreClient()
    supabase_client = SupabaseRerankerClient()
    openai_client = OpenAIRouterClient()

    if clean_before_insert:
        logger.info("Deleting existing records for documents dated between %s and %s...", date_from, date_to)
        supabase_client.delete_by_document_date_range(date_from, date_to)

    logger.info("Retrieving documents from Qdrant for date range %s to %s", date_from, date_to)
    points = qdrant_client.search_points_by_date_range(date_from, date_to)

    logger.info("Generating queries and inserting results...")
    for point in points:
        payload = point.payload
        text = payload.get("combined_text")
        if not text:
            continue

        doc_id = uuid.UUID(point.id)
        doc_date = payload.get("date")  # expected format: YYYY-MM-DD

        try:
            queries = openai_client.generate_queries(text)
            queries_dict = parse_model_json_response(queries)
            rows = prepare_examples(text, doc_id, doc_date, queries_dict)

            for row in rows:
                supabase_client.insert_example(**row)

            logger.info("Inserted %d rows for doc_id: %s", len(rows), doc_id)

        except Exception as e:
            logger.exception("Error processing doc_id %s: %s", doc_id, e)


# run_generation_pipeline('2025-01-01', '2025-01-07')
from datetime import datetime, timedelta


def run_all_weeks_for_2023():
    start_date = datetime.strptime("2025-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2025-06-01", "%Y-%m-%d")

    delta = timedelta(days=7)
    current = start_date

    while current < end_date:
        date_from = current.strftime("%Y-%m-%d")
        date_to = (current + delta).strftime("%Y-%m-%d")
        print(f"\n🚀 Running pipeline from {date_from} to {date_to}")
        try:
            run_generation_pipeline(date_from, date_to, clean_before_insert=False)
        except Exception as e:
            print(f"❌ Failed for range {date_from} to {date_to}: {e}")
        current += delta


run_all_weeks_for_2023()
