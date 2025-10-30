# app/streamlit_app.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from scripts.embeddings import get_embeddings_provider
from scripts.vectorstore import create_or_load_chroma  

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

# ---- Streamlit UI Config ----
st.set_page_config(page_title="Drive RAG Chatbot", layout="centered")
st.title("Drive RAG Chatbot")

# ---- Load embeddings and vector DB ----
embeddings = get_embeddings_provider()
vectordb = create_or_load_chroma(embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ---- Configure OpenRouter LLM (OpenAI-compatible endpoint) ----
llm = init_chat_model("openai:gpt-4o-mini", temperature=0.0)

# ---- Helper: build the prompt manually ----
def build_prompt(question: str, docs: list[Document]) -> str:
    joined = "\n\n".join(d.page_content for d in docs)
    return f"""You are a helpful assistant. Use only the provided context.

Context:
{joined}

Question: {question}

Answer:"""

# ---- Streamlit chat interface ----
query = st.text_input("Ask a question about the Drive documents:")

if query:
    with st.spinner("Retrieving relevant chunks..."):
        docs = retriever.invoke(query)

    if not docs:
        st.warning("No documents found in the vector store. Run the ETL ingest first.")
    else:
        prompt = build_prompt(query, docs)

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)
            answer = getattr(response, "content", str(response))

        st.subheader("Answer:")
        st.write(answer)

