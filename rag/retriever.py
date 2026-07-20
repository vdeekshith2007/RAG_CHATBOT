"""
RAGRetriever — exact port from RAG_first_project/notebook/document.ipynb
Handles query-based retrieval from the ChromaDB vector store.
"""

from typing import List, Dict, Any
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initializes the RAG Retriever.
        Args:
            vector_store: Vector store containing document embeddings.
            embedding_manager: Manager for generating query embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k relevant documents for the given query.
        Args:
            query: The search query.
            top_k: Number of top relevant documents to retrieve.
            score_threshold: Minimum similarity score for retrieved documents.
        Returns:
            List of dicts containing retrieved document and metadata.
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top_k: {top_k}, Score Threshold: {score_threshold}")

        if self.vector_store.get_count() == 0:
            return []

        # Generate embedding for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, self.vector_store.get_count()),
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    similarity_score = 1 - distance  # Convert distance to similarity
                    retrieved_docs.append(
                        {
                            "id": doc_id,
                            "document": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1,
                        }
                    )

                print(f"Retrieved {len(retrieved_docs)} documents.")
            else:
                print("No documents found.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
