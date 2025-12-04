"""
Unified embedding service interface for generating and comparing text embeddings
Automatically switches between local and remote models based on configuration
"""
import numpy as np
from typing import List, Optional
from fastapi import HTTPException

from core.config import settings


class EmbeddingService:
    """
    Unified embedding service that automatically handles local vs remote models
    
    Configuration is done through settings.yaml:
    - model: Model name or path (e.g., "text-embedding-ada-002" or "Alibaba-NLP/gte-multilingual-base")
    - type: "openai" for remote API or "sentence-transformer" for local models
    - dimensions: Embedding dimensions (must match model output)
    
    The service automatically:
    - Detects if model is local or remote based on 'type'
    - Loads local models using sentence-transformers
    - Calls OpenAI API for remote models
    - Exposes consistent get_embedding() interface
    """
    
    def __init__(self, openai_client=None):
        """Initialize embedding service with the configured model"""
        self._model_config = settings.embeddings
        self._model = None
        self._openai_client = openai_client
        
        # Initialize the appropriate model type
        if self._model_config.is_local:
            self._init_local_model()
        elif self._model_config.is_remote:
            self._init_remote_model()
        else:
            raise ValueError(f"Unknown model type: {self._model_config.type}")
        
        print(f"🤖 Embedding service initialized")
        print(f"   Model: {self._model_config.model}")
        print(f"   Type: {self._model_config.type}")
        print(f"   Dimensions: {self._model_config.dimensions}")
    
    def _init_local_model(self):
        """Initialize local model using sentence-transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"📥 Loading local model: {self._model_config.model}")
            self._model = SentenceTransformer(
                self._model_config.model,
                trust_remote_code=True
            )
            print(f"✅ Local model loaded successfully")
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="sentence-transformers not installed. Run: pip install sentence-transformers"
            )
        except Exception as e:
            print(f"❌ Failed to load local model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize local model: {e}"
            )
    
    def _init_remote_model(self):
        """Initialize remote API model"""
        if self._model_config.type == "openai":
            if self._openai_client is None:
                raise ValueError("OpenAI client required for OpenAI models")
            print(f"✅ OpenAI API model configured: {self._model_config.model}")
        else:
            raise ValueError(f"Unknown remote model type: {self._model_config.type}")
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text (unified interface)
        
        Args:
            text: The text to embed
            
        Returns:
            List of float values representing the embedding
        """
        if self._model_config.is_local:
            return self._get_local_embedding(text)
        elif self._model_config.is_remote:
            return self._get_remote_embedding(text)
        else:
            raise ValueError(f"Unknown model type: {self._model_config.type}")
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts (unified interface)
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings, one for each input text
        """
        if self._model_config.is_local:
            return self._get_local_batch_embeddings(texts)
        elif self._model_config.is_remote:
            return self._get_remote_batch_embeddings(texts)
        else:
            raise ValueError(f"Unknown model type: {self._model_config.type}")
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """Get embedding using local sentence-transformer model"""
        try:
            embedding = self._model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            print(f"❌ Local embedding error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get local embedding: {e}"
            )
    
    def _get_local_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings using local sentence-transformer model"""
        try:
            embeddings = self._model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"❌ Local batch embedding error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get local batch embeddings: {e}"
            )
    
    def _get_remote_embedding(self, text: str) -> List[float]:
        """Get embedding using remote API (OpenAI)"""
        if self._model_config.type == "openai":
            try:
                response = self._openai_client.embeddings.create(
                    model=self._model_config.model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"❌ OpenAI embedding error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get OpenAI embedding: {e}"
                )
        else:
            raise ValueError(f"Unknown remote model type: {self._model_config.type}")
    
    def _get_remote_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings using remote API (OpenAI)"""
        if self._model_config.type == "openai":
            try:
                response = self._openai_client.embeddings.create(
                    model=self._model_config.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"❌ OpenAI batch embedding error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get OpenAI batch embeddings: {e}"
                )
        else:
            raise ValueError(f"Unknown remote model type: {self._model_config.type}")
    
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
    
    def compute_holistic_similarity(self, full_answer_embedding: List[float],
                                   key_point_embedding: List[float],
                                   sentence_embeddings: List[List[float]] = None) -> float:
        """
        Compute a holistic similarity score that considers:
        1. Full answer vs key point (captures overall meaning)
        2. Best sentence match (captures specific mentions)
        
        This is more robust than sentence-only matching because it understands
        when concepts are spread across the answer.
        
        Args:
            full_answer_embedding: Embedding of the complete answer
            key_point_embedding: Embedding of the key point
            sentence_embeddings: Optional sentence embeddings for hybrid scoring
            
        Returns:
            Holistic similarity score (0-1)
        """
        # Primary: Full answer to key point similarity
        full_similarity = self.compute_cosine_similarity(full_answer_embedding, key_point_embedding)
        
        if sentence_embeddings:
            # Secondary: Best sentence match
            best_sentence_sim = self.find_best_sentence_similarity(sentence_embeddings, key_point_embedding)
            
            # Combine: Take the MAXIMUM of both approaches
            # This ensures we don't miss when a concept is either:
            # - Explicitly stated in one sentence (best_sentence wins)
            # - Spread across the answer (full_similarity wins)
            combined = max(full_similarity, best_sentence_sim)
            
            # Boost if both approaches agree (high confidence)
            if full_similarity > 0.75 and best_sentence_sim > 0.75:
                combined = min(1.0, combined + 0.05)  # Small boost for agreement
            
            return combined
        
        return full_similarity
    
    @property
    def model_name(self) -> str:
        """Get the current model name"""
        return self._model_config.model
    
    @property
    def dimensions(self) -> int:
        """Get the current embedding dimensions"""
        return self._model_config.dimensions
    
    @property
    def model_type(self) -> str:
        """Get the current model type"""
        return self._model_config.type
