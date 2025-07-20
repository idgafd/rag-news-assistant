from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime, timedelta

from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model as mlflow_load_model
from sentence_transformers.cross_encoder import CrossEncoder

from pipeline.train_challenger import run_challenger_training_pipeline
from pipeline.evaluate import compare_with_production_and_promote
from config.default_config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Reranker API", version="2.0.0")

model = None
client = MlflowClient()
MODEL_NAME = CONFIG["model_name"]


class QueryDocumentPair(BaseModel):
    query: str
    document: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = 10

class RerankResponse(BaseModel):
    results: List[dict]

class ScoreRequest(BaseModel):
    pairs: List[QueryDocumentPair]

class ScoreResponse(BaseModel):
    scores: List[float]

class TrainRequest(BaseModel):
    date_from: Optional[str] = None
    date_to: Optional[str] = None


@app.on_event("startup")
async def load_model():
    global model
    try:
        prod_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        logger.info(f"Loading production model v{prod_version.version}...")
        model = mlflow_load_model(f"models:/{MODEL_NAME}/{prod_version.version}")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load production model: {e}")
        logger.info("Loading base model fallback")
        model = CrossEncoder(CONFIG["base_model"])


@app.get("/")
async def root():
    return {"message": "Document Reranker API", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pairs = [[request.query, doc] for doc in request.documents]
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)

        results = [
            {"document": doc, "score": float(score), "original_index": i}
            for i, (doc, score) in enumerate(zip(request.documents, scores))
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:request.top_k]

        for rank, result in enumerate(results):
            result["rank"] = rank + 1

        return RerankResponse(results=results)

    except Exception as e:
        logger.error(f"Error during reranking: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse)
async def score_pairs(request: ScoreRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        pairs = [[pair.query, pair.document] for pair in request.pairs]
        scores = model.predict(pairs, batch_size=8, show_progress_bar=False)
        return ScoreResponse(scores=[float(score) for score in scores])

    except Exception as e:
        logger.error(f"Error during scoring: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train_challenger/")
async def train_challenger(request: TrainRequest):
    try:
        date_from = request.date_from or (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        date_to = request.date_to or datetime.utcnow().strftime("%Y-%m-%d")

        challenger_info = run_challenger_training_pipeline(date_from=date_from, date_to=date_to, config=CONFIG)

        promoted = compare_with_production_and_promote(
            model_uri=challenger_info["model_uri"],
            eval_dataset=challenger_info["eval_dataset"],
            model_name=MODEL_NAME
        )

        return {
            "status": "completed",
            "challenger_metrics": challenger_info["metrics"],
            "promoted_to_production": promoted
        }
    except Exception as e:
        logger.error(f"Training challenger failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
