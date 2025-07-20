## Architecture

```
Data Collection → Training Data Generation → Model Training → API Serving
     ↓                    ↓                      ↓              ↓
  Qdrant DB         Supabase DB           Local Storage    FastAPI
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11
- 4GB+ RAM

### 1. Clone and Setup

```bash
# Create project directory
mkdir reranker-pipeline
cd reranker-pipeline

# Create required directories
mkdir -p dags logs plugins models/reranker

# Copy all provided files to appropriate locations
# - app.py (FastAPI service)
# - Dockerfile
# - requirements.txt
# - docker-compose.yml
# - dags/data_pipeline_dag.py
# - demo.py
# - other existing scripts: train_reranker.py, generate_training_data.py, parse_raw_batch_data.py
```

### 2. Start Infrastructure

```bash
# Init airflow to start from
docker compose up airflow-init

# Start all services
docker-compose up -d

# Check services
docker-compose ps
```

Services will be available at:
- **Reranker API**: http://localhost:8000
- **Airflow**: http://localhost:8080 (admin/admin)
- **MinIO**: http://localhost:9001 (minioadmin/minioadmin)

### 5. Create `.env` file:
```bash
# Database connections
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_key

# OpenAI for training & data generation
OPENAI_API_KEY=your_openai_key

# Model settings
MODEL_PATH=models/reranker
```

### 4. Test the API

```bash
# Install demo dependencies
pip install requests

# Run demo
python demo.py

# To run into bash firstly create venv
python3 -m venv venv
source venv/bin/activate
pip install requests
```


### 5. Model Training

```bash
# Train model manually
python train_reranker.py \
    --base_model microsoft/MiniLM-L12-H384-uncased \
    --epochs 2 \
    --batch_size 32 \
    --output_dir models/reranker \
    --date_from 2024-01-01 \
    --date_to 2024-12-31
```

### Pipeline Components

#### 1. Data Collection (`parse_raw_batch_data.py`)
- Scrapes articles
- Stores in Qdrant vector database
- Generates embeddings

#### 2. Training Data Generation (`generate_training_data.py`)
- Uses GPT to generate query-document pairs
- Stores in Supabase database
- Creates positive and negative samples

#### 3. Model Training (`train_reranker.py`)
- Trains CrossEncoder model
- Saves model to local storage

#### 4. API Service (`reranker_fast_api_app.py`)
- FastAPI with /rerank and /score endpoints
- Loads trained model
- Serves predictions

### 5. Airflow Pipeline (`dags/data_pipeline_dag.py`)
- Daily data collection
- ~~Weekly model retraining~~ deprecated due to airflow mismatch for this type of tasks
- Automated workflow

### API Usage

#### Rerank Documents

```python
import requests

response = requests.post("http://localhost:8000/rerank", json={
    "query": "machine learning training",
    "documents": [
        "Deep learning models require large datasets",
        "Weather forecast shows rain tomorrow",
        "Neural networks use supervised learning"
    ],
    "top_k": 3
})

print(response.json())
```

#### Score Query-Document Pairs

```python
response = requests.post("http://localhost:8000/score", json={
    "pairs": [
        {
            "query": "AI applications",
            "document": "AI is transforming healthcare industry"
        }
    ]
})

print(response.json())
```

### Container Logs
```bash
# API logs
docker-compose logs -f reranker-api

# Airflow logs
docker-compose logs -f airflow-scheduler
```

### Running Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run training
python train_reranker.py --epochs 1 --batch_size 16
```

### Testing

```bash
# Run demo tests
python demo.py

# Manual API testing
curl -X POST "http://localhost:8000/rerank" \
     -H "Content-Type: application/json" \
     -d '{"query": "test", "documents": ["doc1", "doc2"]}'
```
