"""
VectorStore — exact port from RAG_first_project/notebook/document.ipynb
Uses ChromaDB PersistentClient to store and retrieve document embeddings.
"""

import os
import uuid
import numpy as np
import chromadb
from langchain_core.documents import Document
from typing import List


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(
        self,
        collection_name: str = "chatbot_documents",
        persist_directory: str = "data/vector_store",
    ):
        """
        Initializes the Vector Store.
        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist the vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initializes the ChromaDB client and collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG Chatbot"},
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Adds documents and their embeddings to the vector store.
        Args:
            documents: List of LangChain Document objects.
            embeddings: Numpy array of embeddings corresponding to the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings.")

        print(f"Adding {len(documents)} documents to the vector store...")

        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            # Sanitize metadata: ChromaDB only accepts str, int, float, bool
            sanitized = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    sanitized[k] = v
                else:
                    sanitized[k] = str(v)
            sanitized["doc_index"] = i
            sanitized["content_length"] = len(doc.page_content)
            metadatas.append(sanitized)

            documents_texts.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_texts,
                embeddings=embeddings_list,
            )
            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def get_count(self) -> int:
        """Returns the number of documents in the collection."""
        return self.collection.count() if self.collection else 0

    def reset(self):
        """Deletes and recreates the collection (clears all documents)."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Document embeddings for RAG Chatbot"},
            )
            print("Vector store reset successfully.")
        except Exception as e:
            print(f"Error resetting vector store: {e}")
            raise
