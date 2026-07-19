"""
Flask Backend for RAG_Chatbot
Serves the frontend and exposes API endpoints for document upload and querying.
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore
from rag.retriever import RAGRetriever
from rag.pipeline import get_llm, rag_advanced
from loaders.document_loader import load_document, SUPPORTED_EXTENSIONS

# ─────────────────────────────────────────────────────────────
# Flask App Configuration
# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vector_store")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max

# ─────────────────────────────────────────────────────────────
# Initialize RAG components (singleton, loaded once at startup)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("Initializing RAG Chatbot...")
print("=" * 60)

embedding_manager = EmbeddingManager()
vector_store = VectorStore(persist_directory=VECTOR_STORE_DIR)
retriever = RAGRetriever(vector_store, embedding_manager)

# LLM is lazy-initialized once and uses GROQ_API_KEY from .env
_llm = None


def get_cached_llm():
    """Returns cached LLM or creates new one."""
    global _llm

    if _llm is None:
        _llm = get_llm()
    return _llm


print("RAG Chatbot initialized successfully!\n")


# ─────────────────────────────────────────────────────────────
# Routes — Frontend
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


# ─────────────────────────────────────────────────────────────
# API — Status
# ─────────────────────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def status():
    """Returns the current status of the vector store."""
    count = vector_store.get_count()
    return jsonify(
        {
            "status": "ready",
            "documents_loaded": count,
            "has_documents": count > 0,
        }
    )


# ─────────────────────────────────────────────────────────────
# API — Upload Document
# ─────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_document():
    """
    Accepts a document file, loads it, generates embeddings, and stores in ChromaDB.
    Supported: PDF, TXT, DOC, DOCX, Excel (xlsx/xls), CSV
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Attempting to load uncommon file type: {ext}")

    # Save uploaded file
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        # Load document — same loaders as notebook
        documents = load_document(file_path)

        if not documents:
            return jsonify({"error": "Could not extract text from the document"}), 400

        # Keep only the current upload in the vector store.
        # This removes previously indexed chunks before inserting the new ones.
        vector_store.reset()

        # Generate embeddings — same EmbeddingManager as notebook
        texts = [doc.page_content for doc in documents]
        embeddings = embedding_manager.generate_embeddings(texts)

        # Store in ChromaDB — same VectorStore as notebook
        vector_store.add_documents(documents, embeddings)

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "chunks_loaded": len(documents),
                "total_documents": vector_store.get_count(),
                "message": f"Successfully loaded '{filename}' with {len(documents)} chunk(s). Previous document was cleared.",
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)


# ─────────────────────────────────────────────────────────────
# API — Query
# ─────────────────────────────────────────────────────────────
@app.route("/api/query", methods=["POST"])
def query():
    """
    Accepts a user query and returns an answer using the RAG pipeline.
    Body: { "query": "...", "top_k": 5 (optional) }
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "No query provided"}), 400

    user_query = data["query"].strip()
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    if vector_store.get_count() == 0:
        return jsonify(
            {
                "error": "No documents loaded. Please upload a document first.",
                "answer": None,
            }
        ), 400

    if not os.getenv("GROQ_API_KEY"):
        return jsonify(
            {
                "error": "GROQ_API_KEY not found. Set it in .env."
            }
        ), 400

    top_k = int(data.get("top_k", 5))

    try:
        llm = get_cached_llm()

        # Run rag_advanced() — exact port from notebook
        result = rag_advanced(
            query=user_query,
            retriever=retriever,
            llm=llm,
            top_k=top_k,
            min_score=0.2,
            return_context=False,
        )

        return jsonify(
            {
                "answer": result["answer"],
                "sources": result["sources"],
                "confidence": result["confidence"],
                "query": user_query,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# API — Reset Vector Store
# ─────────────────────────────────────────────────────────────
@app.route("/api/reset", methods=["DELETE"])
def reset():
    """Clears all documents from the vector store."""
    try:
        vector_store.reset()
        return jsonify(
            {"success": True, "message": "Vector store cleared. Ready for new documents."}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nStarting RAG Chatbot server...")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
