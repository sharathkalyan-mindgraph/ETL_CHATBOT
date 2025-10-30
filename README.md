
# Drive RAG Chatbot – End-to-End ETL and Retrieval-Augmented Generation Pipeline

## Overview
The **Drive RAG Chatbot** is an end-to-end Retrieval-Augmented Generation (RAG) application that extracts documents from Google Drive, processes them into embeddings, and allows natural language querying through a Streamlit chat interface.

It integrates:
- **Google Drive API** for document ingestion  
- **LangChain** for embeddings and vector retrieval  
- **ChromaDB** for vector storage  
- **HuggingFace Embeddings** for semantic text representation  
- **Tesseract OCR** for extracting text from images  
- **Streamlit** for an interactive chatbot interface  
- **OpenRouter/OpenAI-compatible LLMs** for generating answers  

---

## Project Structure

```
ETL_CBT/
├── app/
│   ├── streamlit_app.py          # Streamlit-based chatbot interface
│
├── scripts/
│   ├── etl.py                    # File text extraction and chunking
│   ├── drive_utils.py            # Google Drive authentication & file operations
│   ├── embeddings.py             # Embedding model initialization
│   ├── vectorstore.py            # ChromaDB creation and updates
│   ├── etl_runner.py             # Main ETL orchestration script
│
├── data/
│   └── chroma_db/                # Persisted Chroma vector store
│
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
├── service-account.json          # Google Drive API credentials (not committed)
└── README.md                     # Documentation (this file)
```

---

## 1. Prerequisites

Before starting, ensure you have:

- **Python 3.10+**
- **Google Cloud Project** with Drive API enabled
- **Tesseract OCR** installed locally  
  - Windows path (default):  
    ```
    C:\Program Files\Tesseract-OCR\tesseract.exe
    ```
  - Add this path to your environment if necessary.

---

## 2. Installation Steps

### Step 1: Clone the Repository
```bash
git clone https://github.com/sharathkalyan-mindgraph/ETL_CHATBOT.git
cd ETL_CHATBOT
```

### Step 2: Create and Activate a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate   # On Linux/Mac
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables
Create a `.env` file in the root directory with the following content:

```ini
# Google Drive API
SERVICE_ACCOUNT_FILE=./service-account.json
DRIVE_FOLDER_ID=your_google_drive_folder_id

# Vector Store
CHROMA_DB_PATH=./data/chroma_db

# Embeddings
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# OpenRouter / LLM Configuration
OPENAI_API_KEY=your_openrouter_api_key
```

### Step 5: Add Google Service Account File
Download your Google Drive **Service Account JSON key** and place it in the root folder as:
```
./service-account.json
```

---

## 3. Running the ETL Pipeline

The ETL process performs the following:
- Connects to your Google Drive folder
- Downloads all supported documents locally (PDF, DOCX, TXT, PNG, JPEG)
- Extracts text using pdfplumber / docx / Tesseract OCR
- Splits text into semantic chunks
- Converts chunks into embeddings using HuggingFace
- Stores them in Chroma vector database

To run it:
```bash
python -m scripts.etl_runner
```

You should see logs like:
```
[INFO] Starting ETL process...
[INFO] File downloaded successfully to: C:\Users\...\Temp\node.png
[INFO] OCR extracted 2209 characters from node.png
[INFO] Split node.png into 3 chunks.
[INFO] Ingested 84 new chunks into Chroma.
[INFO] ETL process complete.
```

---

## 4. Running the Streamlit Chatbot

Once the ETL step is complete, launch the chatbot UI:
```bash
streamlit run app/streamlit_app.py
```

Open your browser and navigate to:
```
http://localhost:8501
```

Ask questions about your Drive documents directly (e.g.):
> “Explain what Node.js event-driven model means.”  
> “What is the difference between RNN and LSTM architectures?”

The chatbot retrieves the most relevant chunks and generates contextual answers using the LLM.

---

## 5. How It Works (Architecture)

### Pipeline Flow

1. **Google Drive Layer**  
   - Authenticates with a service account  
   - Lists and downloads files from the target folder  

2. **ETL Layer**  
   - Extracts text from each file (OCR for images, pdfplumber for PDFs, etc.)
   - Splits long texts into overlapping chunks  

3. **Vectorization Layer**  
   - Converts text chunks into dense embeddings using HuggingFace  
   - Stores them in ChromaDB for semantic retrieval  

4. **RAG Chat Interface**  
   - User queries processed via Streamlit  
   - Relevant chunks fetched from Chroma  
   - LLM generates answers based only on retrieved context  

---

## 6. Supported File Types
| File Type | Extraction Method        | Notes |
|------------|--------------------------|--------|
| `.pdf`     | pdfplumber               | Text only |
| `.docx`    | python-docx              | Paragraph-based |
| `.png`, `.jpg`, `.jpeg` | Tesseract OCR | Uses OpenCV preprocessing |
| `.txt`     | UTF-8 text read          | Fallback |

---

## 7. Customization

- Change embedding model:
  ```env
  LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L12-v2
  ```
- Change number of retrieved chunks:
  Edit in `app/streamlit_app.py`:
  ```python
  retriever = vectordb.as_retriever(search_kwargs={"k": 4})
  ```

---

## 8. Troubleshooting

| Issue | Cause | Fix |
|-------|--------|-----|
| `No text detected by OCR` | Image too blurry / small | Preprocess or upscale the image |
| `ModuleNotFoundError: scripts.drive_utils` | Wrong working directory | Run scripts from project root |
| `Permission denied for Drive` | Invalid service account permissions | Share Drive folder with your service account email |
| `LangChainDeprecationWarning` | Old embedding import | Safe to ignore or upgrade to `langchain-huggingface` |

---

## 9. Future Improvements

- Streamlined in-memory Drive file reading (no temp file writes)  
- Vector store cleanup & re-ingest options  
- Optional image captioning for better OCR context  
- Cloud-hosted deployment on Render / HuggingFace Spaces  
- Automating the embeddings store
---
