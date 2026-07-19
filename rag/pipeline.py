"""
RAG Pipeline — rag_advanced() ported from RAG_first_project/notebook/document.ipynb
Uses Groq LLM (llama-3.1-8b-instant) to generate answers from retrieved context.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from rag.retriever import RAGRetriever

load_dotenv()


def get_llm() -> ChatGroq:
    """
    Initializes and returns the Groq LLM.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("GROQ_API_KEY not found. Set it in .env.")
    return ChatGroq(
        groq_api_key=key,
        model_name="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm: ChatGroq,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> dict:
    """
    Enhanced RAG pipeline — exact port from notebook rag_advanced().
    Returns answer, sources, confidence score, and optionally the context.

    Args:
        query: User question.
        retriever: RAGRetriever instance.
        llm: ChatGroq LLM instance.
        top_k: Number of docs to retrieve.
        min_score: Minimum similarity score threshold.
        return_context: Whether to include retrieved context in output.

    Returns:
        dict with keys: answer, sources, confidence, [context]
    """
    # Retrieve the context
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            "answer": "No relevant context found in the uploaded documents. Please upload a document first.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    context = "\n\n".join([doc["document"] for doc in results])

    sources = [
        {
            "source": doc["metadata"].get(
                "source_file", doc["metadata"].get("source", "unknown")
            ),
            "page": doc["metadata"].get("page", "N/A"),
            "score": round(doc["similarity_score"], 4),
            "preview": doc["document"][:120] + "...",
        }
        for doc in results
    ]

    confidence = max([doc["similarity_score"] for doc in results])

    prompt = f"""Use the following context to answer the question concisely.

Context:
{context}

Question: {query}
Answer: """

    response = llm.invoke([prompt])

    output = {
        "answer": response.content,
        "sources": sources,
        "confidence": round(confidence, 4),
    }

    if return_context:
        output["context"] = context

    return output
