# RAG News Assistant

[🔗 Launch the App](https://rag-news-assistant-fpbvl4ndjsx2vrhjmtmsxy.streamlit.app/)

RAG News Assistant is your curious, slightly sarcastic AI sidekick for exploring the world of tech news. Just ask a question — or throw in an image for extra info — and it’ll fetch the most relevant articles, summarize them, and serve you a snappy answer.



## What It Can Do

* Understands natural language queries (like "what happened in AI last month?")
* Accepts optional image input and retrieves visually similar articles
* Searches through a vector database of tech news (powered by Qdrant)
* Uses GPT-4 Turbo to answer your question based on retrieved info
* Falls back to GPT’s general knowledge if no article is relevant



## How to Get It Running

### 1. Clone the project

```bash
git clone https://github.com/idgafd/rag-news-assistant.git
cd rag-news-assistant
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Add your secret keys

Create a file called `.streamlit/secrets.toml` or `.env` and add:

```toml
QDRANT_API_KEY = "your_qdrant_key"
QDRANT_URL = "your_qdrant_url"
QDRANT_COLLECTION = "batch_articles"
OPENAI_API_KEY = "your_openai_key"
```

### 5. Parse the data

Run `batch_parser.py` with necessary date filters to fetch relevant weekly news



## Project Structure

```
rag-news-assistant/
├── app.py                   # Main Streamlit interface
├── core_pipeline.py         # Connects routing + search + answering
├── openai_wrapper.py        # All OpenAI calls (routing / answering)
├── qdrant_client_wrapper.py # Qdrant client and queries
├── batch_parser.py          # Parses and ingests articles
├── model_registry.py        # Loads embedding and caption models
├── config.py                # Secret loader
├── utils.py                 # Extra functions
├── prompts/                 # All system prompts
└── requirements.txt
```



## Run It Locally

```bash
streamlit run app.py
```

Or try the local `core_pipeline.py` with help of example usage



## Tech Behind the Scenes

* **Streamlit** for UI
* **OpenAI GPT-4 Turbo** for query routing and responses
* **Qdrant** as the vector search engine
* **SentenceTransformers + CLIP + BLIP** for text and image embeddings
* **BeautifulSoup** for adding new scraped sources



## Notes

* Articles are from [The Batch](https://www.deeplearning.ai/the-batch/)
* Combined text + image caption embeddings power retrieval
* If nothing is found — GPT jumps in with a general answer


