import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# def get_config(key: str) -> str:
#     return st.secrets.get(key, os.getenv(key))

def get_config(key: str) -> str:
    return os.getenv(key)


QDRANT_API_KEY = get_config("QDRANT_API_KEY")
QDRANT_URL = get_config("QDRANT_URL")
QDRANT_COLLECTION = get_config("QDRANT_COLLECTION")
CLUSTER_ID = get_config("CLUSTER_ID")
OPENAI_API_KEY = get_config("OPENAI_API_KEY")
SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")
