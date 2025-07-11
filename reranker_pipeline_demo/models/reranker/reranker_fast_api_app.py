from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import logging
from sentence_transformers.cross_encoder import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Document Reranker API", version="1.0.0")

model = None


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


@app.on_event("startup")
async def load_model():
    global model
    model_path = os.getenv("MODEL_PATH", "models/reranker")

    try:
        if (os.path.exists(model_path)
                and os.path.exists(os.path.join(model_path, "config.json"))
                and os.path.exists(os.path.join(model_path, "pytorch_model.bin"))):
            logger.info(f"Loading trained model from {model_path}")
            model = CrossEncoder(model_path)
        else:
            logger.info("No trained model found, using base model")
            model = CrossEncoder("microsoft/MiniLM-L12-H384-uncased")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    return {"message": "Document Reranker API", "status": "healthy"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on query relevance
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.documents:
        raise HTTPException(status_code=400, detail="No documents provided")

    try:
        # Create query-document pairs
        pairs = [[request.query, doc] for doc in request.documents]

        # Get relevance scores
        scores = model.predict(pairs)

        # Create results with scores and rankings
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            results.append({
                "document": doc,
                "score": float(score),
                "original_index": i
            })

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Limit to top_k
        results = results[:request.top_k]

        # Add ranking
        for rank, result in enumerate(results):
            result["rank"] = rank + 1

        return RerankResponse(results=results)

    except Exception as e:
        logger.error(f"Error during reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score", response_model=ScoreResponse)
async def score_pairs(request: ScoreRequest):
    """
    Score query-document pairs
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.pairs:
        raise HTTPException(status_code=400, detail="No pairs provided")

    try:
        # Convert to format expected by model
        pairs = [[pair.query, pair.document] for pair in request.pairs]

        # Get scores
        scores = model.predict(pairs)

        return ScoreResponse(scores=[float(score) for score in scores])

    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)