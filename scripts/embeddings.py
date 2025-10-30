# scripts/embeddings.py
import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings_provider():
    """
    Initialize and return the embeddings provider.
    Falls back gracefully if model loading fails.
    """
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    try:
        print(f"[INFO] Loading embeddings model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        print("[INFO] Embeddings model loaded successfully.")
        return embeddings

    except Exception as e:
        print(f"[ERROR] Failed to load embeddings model '{model_name}': {e}")

        # Attempt fallback to a lightweight default model
        fallback_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        try:
            print(f"[WARN] Attempting fallback embeddings model: {fallback_model}")
            embeddings = HuggingFaceEmbeddings(model_name=fallback_model)
            print("[INFO] Fallback embeddings model loaded successfully.")
            return embeddings
        except Exception as e2:
            print(f"[CRITICAL] Could not load fallback embeddings model either: {e2}")
            return None
