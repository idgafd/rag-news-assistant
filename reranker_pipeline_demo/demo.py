import requests

# API endpoint
API_URL = "http://localhost:8000"


def test_health():
    """Test API health"""
    print("Testing API health...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_reranking():
    """Test document reranking"""
    print("Testing document reranking...")

    # Sample query and documents
    query = "machine learning model training"
    documents = [
        "Deep learning models require large datasets for effective training and validation.",
        "The weather forecast shows rain tomorrow with temperatures around 20 degrees.",
        "Neural networks can be trained using supervised learning techniques with labeled data.",
        "Cooking pasta requires boiling water and adding salt for better flavor.",
        "Cross-validation is an important technique for evaluating model performance during training.",
        "The stock market showed significant volatility last week with major indices declining.",
        "Transfer learning allows pre-trained models to be fine-tuned on specific tasks.",
        "Social media platforms are implementing new privacy policies this year."
    ]

    # Make API request
    request_data = {
        "query": query,
        "documents": documents,
        "top_k": 5
    }

    response = requests.post(f"{API_URL}/rerank", json=request_data)

    if response.status_code == 200:
        results = response.json()
        print(f"Reranking successful!")
        print(f"Query: {query}")
        print("\nTop ranked documents:")
        for i, result in enumerate(results["results"]):
            print(f"\n{i + 1}. [Score: {result['score']:.4f}]")
            print(f"   Document: {result['document'][:100]}...")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    print()


def test_scoring():
    """Test query-document pair scoring"""
    print("Testing query-document scoring...")

    # Sample query-document pairs
    pairs = [
        {
            "query": "artificial intelligence applications",
            "document": "AI is transforming various industries including healthcare, finance, and transportation."
        },
        {
            "query": "artificial intelligence applications",
            "document": "The recipe for chocolate cake requires flour, sugar, eggs, and cocoa powder."
        },
        {
            "query": "data science techniques",
            "document": "Statistical analysis and machine learning are core components of data science."
        }
    ]

    request_data = {"pairs": pairs}

    response = requests.post(f"{API_URL}/score", json=request_data)

    if response.status_code == 200:
        results = response.json()
        print(f"Scoring successful!")
        print("Query-Document relevance scores:")
        for i, (pair, score) in enumerate(zip(pairs, results["scores"])):
            print(f"\n{i + 1}. [Score: {score:.4f}]")
            print(f"   Query: {pair['query']}")
            print(f"   Document: {pair['document'][:80]}...")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    print()


def demo_use_case():
    """Demonstrate a real-world use case"""
    print("Real-world demo: Finding relevant AI articles...")

    query = "transformer architecture deep learning"

    ai_articles = [
        "The Transformer architecture revolutionized natural language processing with self-attention mechanisms.",
        "Climate change is causing unprecedented shifts in global weather patterns.",
        "BERT and GPT models are based on the Transformer architecture and have achieved state-of-the-art results.",
        "Investment strategies for cryptocurrency markets require careful risk assessment.",
        "Attention mechanisms in neural networks help models focus on relevant input features.",
        "Renewable energy sources are becoming more cost-effective than fossil fuels.",
        "Vision Transformers adapt the Transformer architecture for computer vision tasks.",
        "Urban planning considers factors like population density and transportation infrastructure."
    ]

    request_data = {
        "query": query,
        "documents": ai_articles,
        "top_k": 3
    }

    response = requests.post(f"{API_URL}/rerank", json=request_data)

    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results['results'])} most relevant articles:")
        print(f"Search query: '{query}'")
        print("\nRelevant articles:")
        for i, result in enumerate(results["results"]):
            print(f"\n🏆 Rank {result['rank']} (Score: {result['score']:.4f})")
            print(f"   {result['document']}")
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")
    print()


def main():
    """Run all demo tests"""
    print("Document Reranker API Demo")
    print("=" * 50)

    try:
        test_health()
        test_reranking()
        test_scoring()
        demo_use_case()

        print("All tests completed successfully!")

    except requests.exceptions.ConnectionError:
        print("Could not connect to API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
