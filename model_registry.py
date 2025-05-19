import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, CrossEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

text_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
rerank_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
