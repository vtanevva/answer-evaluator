"""
OpenAI embedding service for generating and comparing text embeddings
"""
import numpy as np
from typing import List, Dict
from fastapi import HTTPException

from core.config import settings


class EmbeddingService:
    """
    Service for generating and managing text embeddings using OpenAI API
    
    This service handles:
    - Generating embeddings
    - Computing cosine similarity between embeddings
    - Batch embedding generation for efficiency
    """
    
    def __init__(self, openai_client):
        """Initialize embedding service with configuration"""
        self._openai_client = openai_client
        self._model_name = settings.openai.model_name
        self._embedding_dimensions = settings.openai.embedding_dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using OpenAI text-embedding-ada-002
        
        This converts text into a high-dimensional vector that represents
        the semantic meaning of the text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding
            
        Raises:
            HTTPException: If embedding generation fails
        """
        try:
            response = self._openai_client.embeddings.create(
                model=self._model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Error getting embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get embedding for text: {text}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in a single API call for efficiency
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings corresponding to input texts
            
        Raises:
            HTTPException: If embedding generation fails
        """
        try:
            response = self._openai_client.embeddings.create(
                model=self._model_name,
                input=texts
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"❌ Error getting batch embeddings: {e}")
            raise HTTPException(status_code=500, detail="Failed to get batch embeddings")
    
    def compute_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors
        
        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (perfect semantic match)
        - 0.0 = perpendicular (no semantic similarity)
        - -1.0 = opposite direction (semantically opposite)
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between -1 and 1
        """
        # Convert to numpy arrays for efficient computation
        vector_a = np.array(embedding1)
        vector_b = np.array(embedding2)
        
        # Compute dot product
        dot_product = np.dot(vector_a, vector_b)
        
        # Compute vector magnitudes
        magnitude_a = np.linalg.norm(vector_a)
        magnitude_b = np.linalg.norm(vector_b)
        
        # Avoid division by zero
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        # Return cosine similarity
        return dot_product / (magnitude_a * magnitude_b)
    
    def find_best_sentence_similarity(self, sentence_embeddings: List[List[float]], 
                                    key_point_embedding: List[float]) -> float:
        """
        Find the highest similarity between any sentence and a key point
        
        Args:
            sentence_embeddings: List of sentence embeddings from user answer
            key_point_embedding: Embedding of the key point to compare against
            
        Returns:
            Maximum similarity score found
        """
        if not sentence_embeddings:
            return 0.0
            
        similarities = [
            self.compute_cosine_similarity(sentence_emb, key_point_embedding)
            for sentence_emb in sentence_embeddings
        ]
        
        return max(similarities)
