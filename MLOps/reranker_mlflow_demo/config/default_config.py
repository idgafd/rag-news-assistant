CONFIG = {
    "model_name": "CrossEncoderReranker",
    "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "experiment_name": "ChallengerTraining",
    "output_dir": "models/reranker",
    "epochs": 2,
    "batch_size": 8,
    "num_negatives": 4
}