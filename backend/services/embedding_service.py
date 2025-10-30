"""
Multi-provider embedding service for generating and comparing text embeddings
Supports OpenAI and GTE-multilingual models with easy switching
"""
import numpy as np
from typing import List, Dict, Optional
from fastapi import HTTPException
import torch
from abc import ABC, abstractmethod

from core.config import settings


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        pass
    
    @abstractmethod
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name"""
        pass
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get embedding dimensions"""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(self, openai_client):
        self._client = openai_client
        self._model_name = settings.embeddings.openai_model
        self._dimensions = settings.embeddings.openai_dimensions
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API"""
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ OpenAI embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get OpenAI embedding: {e}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings using OpenAI API"""
        try:
            response = self._client.embeddings.create(
                model=self._model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"❌ OpenAI batch embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get OpenAI batch embeddings: {e}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dimensions(self) -> int:
        return self._dimensions


class GTEMultilingualProvider(EmbeddingProvider):
    """GTE-multilingual embedding provider using sentence-transformers"""
    
    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                settings.embeddings.gte_model,
                trust_remote_code=True
            )
            self._model_name = settings.embeddings.gte_model
            self._dimensions = settings.embeddings.gte_dimensions
            print(f"✅ Loaded GTE-multilingual model: {self._model_name}")
        except Exception as e:
            print(f"❌ Failed to load GTE-multilingual model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize GTE-multilingual: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using GTE-multilingual model"""
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ GTE-multilingual embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get GTE embedding: {e}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings using GTE-multilingual model"""
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"❌ GTE-multilingual batch embedding error: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to get GTE batch embeddings: {e}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dimensions(self) -> int:
        return self._dimensions


class EmbeddingService:
    """
    Multi-provider embedding service that supports easy switching between providers
    
    To switch providers, simply change the 'provider' setting in settings.yaml:
    - "openai" for OpenAI text-embedding-ada-002
    - "gte-multilingual" for Alibaba GTE-multilingual-base
    """
    
    def __init__(self, openai_client=None):
        """Initialize embedding service with the configured provider"""
        self._provider = self._create_provider(openai_client)
        print(f"🤖 Embedding service initialized with: {self._provider.model_name}")
        print(f"📏 Embedding dimensions: {self._provider.dimensions}")
    
    def _create_provider(self, openai_client) -> EmbeddingProvider:
        """Create the appropriate embedding provider based on configuration"""
        provider_name = settings.embeddings.provider.lower()
        
        if provider_name == "openai":
            if openai_client is None:
                raise ValueError("OpenAI client required for OpenAI embedding provider")
            return OpenAIEmbeddingProvider(openai_client)
        
        elif provider_name == "gte-multilingual":
            return GTEMultilingualProvider()
        
        else:
            raise ValueError(f"Unknown embedding provider: {provider_name}. Supported: 'openai', 'gte-multilingual'")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text using the configured provider
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding
        """
        return self._provider.get_embedding(text)
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts using the configured provider
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, one for each input text
        """
        return self._provider.get_batch_embeddings(texts)
    
    def compute_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors
        
        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (perfect match)
        - 0.0 = perpendicular (no similarity)
        - -1.0 = opposite direction (opposite meaning)
        
        Formula: cos(θ) = (A · B) / (||A|| × ||B||)
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
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
        
        # Compute cosine similarity
        cosine_similarity = dot_product / (magnitude_a * magnitude_b)
        
        # Ensure the result is within [-1, 1] due to floating point errors
        return np.clip(cosine_similarity, -1.0, 1.0)
    
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
    
    @property
    def model_name(self) -> str:
        """Get the current model name"""
        return self._provider.model_name
    
    @property
    def dimensions(self) -> int:
        """Get the current embedding dimensions"""
        return self._provider.dimensions
    
    @property
    def provider_name(self) -> str:
        """Get the current provider name"""
        return settings.embeddings.provider
