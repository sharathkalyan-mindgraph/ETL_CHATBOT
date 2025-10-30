# scripts/vectorstore.py
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Default Chroma database directory
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./data/chroma_db")


def create_chroma_from_chunks(chunks, embeddings):
    """
    Create or update a Chroma vector database from extracted document chunks.
    Each chunk is expected to be a dict with keys: 'text' and 'metadata'.
    """
    try:
        if not chunks or not isinstance(chunks, list):
            print("[ERROR] Invalid input: 'chunks' must be a non-empty list.")
            return None

        docs = []
        for c in chunks:
            try:
                text = c.get("text", "").strip()
                metadata = c.get("metadata", {})
                if text:
                    docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                print(f"[WARN] Skipping malformed chunk: {e}")

        if not docs:
            print("[WARN] No valid document chunks to index.")
            return None

        splitted_docs = []
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            for d in docs:
                for s in splitter.split_text(d.page_content):
                    splitted_docs.append(Document(page_content=s, metadata=d.metadata))
        except Exception as e:
            print(f"[ERROR] Failed to split documents: {e}")
            splitted_docs = docs 


        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            vectordb = Chroma.from_documents(
                documents=splitted_docs,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
            vectordb.persist()
            print(f"[INFO] Chroma vector store created/updated at: {CHROMA_DIR}")
            return vectordb

        except Exception as e:
            print(f"[ERROR] Failed to create/update Chroma vector store: {e}")
            return None

    except Exception as e:
        print(f"[CRITICAL] Unexpected error in create_chroma_from_chunks: {e}")
        return None


def create_or_load_chroma(embeddings):
    """
    Load the existing Chroma DB if it exists, otherwise create a new empty one.
    """
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
    except Exception as e:
        print(f"[ERROR] Could not create directory {CHROMA_DIR}: {e}")
        return None

    try:
        vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        print(f"[INFO] Loaded Chroma DB from {CHROMA_DIR}")
        return vectordb

    except Exception as e:
        print(f"[WARN] Could not load existing Chroma DB: {e}")
        print("[INFO] Creating a new empty vector store as fallback.")
        try:
            vectordb = Chroma.from_documents([], embedding=embeddings, persist_directory=CHROMA_DIR)
            vectordb.persist()
            return vectordb
        except Exception as e2:
            print(f"[CRITICAL] Failed to create fallback Chroma DB: {e2}")
            return None
