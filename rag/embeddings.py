"""
EmbeddingManager — exact port from RAG_first_project/notebook/document.ipynb
Uses SentenceTransformer("all-MiniLM-L6-v2") to generate 384-dim embeddings.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingManager:
    """Handles document embedding generation using Sentence Transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Embedding Manager.
        Args:
            model_name: HuggingFace model name for sentence embedding.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the sentence transformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generates embeddings for a list of texts.
        Args:
            texts: List of strings to embed.
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dimension).
        """
        if not self.model:
            raise ValueError("Model not loaded.")
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        print(f"Embeddings generated successfully. Shape: {embeddings.shape}")
        return embeddings
