# RAG Chatbot

A document-question-answering chatbot built around a traditional RAG pipeline.
It mirrors the embedding, ChromaDB, retrieval, and Groq generation flow from `RAG_first_project/notebook/document.ipynb`.

## Features

- **Multi-format Document Support**: PDF, TXT, DOC/DOCX, Excel (xlsx/xls), CSV, plus a fallback for other common document formats
- **Exact RAG Pipeline** from the notebook:
  - `SentenceTransformer("all-MiniLM-L6-v2")` for embeddings (384-dim)
  - `ChromaDB` PersistentClient for vector storage
  - Custom `RAGRetriever` with cosine similarity retrieval
  - `rag_advanced()` with Groq LLM (llama-3.1-8b-instant)
- **Simple Web UI** for upload, chat, and source inspection
- **Flask REST API** with 4 endpoints

## Project Structure

```
RAG_Chatbot/
├── app.py                  # Flask server
├── .env                    # GROQ_API_KEY
├── requirements.txt
├── rag/
│   ├── embeddings.py       # EmbeddingManager
│   ├── vector_store.py     # VectorStore (ChromaDB)
│   ├── retriever.py        # RAGRetriever
│   └── pipeline.py         # rag_advanced() + Groq LLM
├── loaders/
│   └── document_loader.py  # Multi-format loader
├── data/
│   └── vector_store/       # ChromaDB persistence
├── uploads/                # Temporary upload storage
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Groq API Key

Edit `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```

Or enter it directly in the UI sidebar.

Get a free API key at: https://console.groq.com

### 3. Run the Server

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

If you upload a file type that is not listed explicitly, the backend still tries to parse it through the unstructured fallback loader.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Server status & document count |
| POST | `/api/upload` | Upload document (multipart/form-data) |
| POST | `/api/query` | Ask a question (JSON body) |
| DELETE | `/api/reset` | Clear all documents |

## RAG Pipeline (same as notebook)

1. **Load** → PyMuPDFLoader / TextLoader / Docx2txtLoader / CSVLoader / pandas Excel
2. **Embed** → `SentenceTransformer("all-MiniLM-L6-v2")` → 384-dim vectors
3. **Store** → ChromaDB `PersistentClient` collection
4. **Retrieve** → `RAGRetriever.retrieve()` → top-k cosine similarity
5. **Generate** → `ChatGroq(llama-3.1-8b-instant)` with context prompt
