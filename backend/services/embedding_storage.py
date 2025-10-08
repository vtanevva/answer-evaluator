import json
import os
import time
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec

from core.config import settings


class EmbeddingStorage:
    """
    Service for persisting and loading precomputed embeddings to/from Pinecone
    
    This service handles:
    - Creating and managing Pinecone index
    - Upserting embeddings with metadata to Pinecone
    - Querying embeddings from Pinecone
    - Managing embedding metadata (timestamps, question hashes)
    """
    
    def __init__(self):
        """
        Initialize embedding storage service with Pinecone
        """
        self._pinecone_config = settings.pinecone
        
        # Fallback to environment variable if api_key is not set in settings
        if not self._pinecone_config.api_key:
            self._pinecone_config.api_key = os.getenv("PINECONE_API_KEY", "")

        if not self._pinecone_config.api_key:
            raise ValueError("Pinecone API key not found in configuration or environment")
        
        # Initialize Pinecone client
        self._pc = Pinecone(api_key=self._pinecone_config.api_key)
        
        # Get the correct dimensions from embedding config
        self._embedding_dimensions = settings.embeddings.current_dimensions
        
        # Build an index name that encodes provider and dimension to avoid conflicts
        base_name = self._pinecone_config.index_name
        provider = settings.embeddings.provider.replace(" ", "-").lower()
        self._index_name = f"{base_name}-{provider}-{self._embedding_dimensions}"

        # Initialize or get the index
        self._index = self._get_or_create_index()
    
    def _get_or_create_index(self):
        """
        self._file_path = file_path or os.path.join(
            os.path.dirname(__file__), 
            "..", 
            settings.grading.embeddings_file_path
        )
    
    def cache_embeddings(
        self, 
        key_point_embeddings: Dict[int, List[List[float]]], 
        key_point_keywords: Dict[int, List[Set[str]]],
        questions_metadata: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Save precomputed embeddings and keywords to Pinecone
        
        Args:
            key_point_embeddings: Dictionary mapping question_id to list of embeddings
            key_point_keywords: Dictionary mapping question_id to list of keyword sets
            questions_metadata: Metadata about questions for validation
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            vectors_to_upsert = []
            
            for question_id, embeddings in key_point_embeddings.items():
                question_keywords = key_point_keywords.get(question_id, [])
                question_meta = questions_metadata.get(question_id, {})
                
                for idx, embedding in enumerate(embeddings):
                    # Create unique ID for each key point embedding
                    vector_id = f"q{question_id}_kp{idx}"
                    
                    # Prepare metadata
                    metadata = {
                        "question_id": question_id,
                        "key_point_index": idx,
                        "question_text": question_meta.get("question_text", ""),
                        "key_points_count": question_meta.get("key_points_count", 0),
                        "keywords": list(question_keywords[idx]) if idx < len(question_keywords) else [],
                        "created_at": datetime.now().isoformat(),
                        "embedding_provider": settings.embeddings.provider,
                        "embedding_model": settings.embeddings.current_model
                    }
                    
                    vectors_to_upsert.append({
                        "id": vector_id,
                        "values": embedding,
                        "metadata": metadata
                    })
            
            # Upsert vectors in batches (Pinecone recommends batch size of 100)
            batch_size = 100
            total_vectors = len(vectors_to_upsert)
            
            print(f"🔄 Upserting {total_vectors} vectors to Pinecone...")
            
            for i in range(0, total_vectors, batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self._index.upsert(vectors=batch)
                print(f"   📦 Batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size} uploaded")
            
            # Wait for upserts to be processed
            time.sleep(2)
            
            # Verify upload
            stats = self._index.describe_index_stats()
            print(f"✅ Successfully cached embeddings to Pinecone")
            print(f"   📊 Total vectors in index: {stats['total_vector_count']}")
            print(f"   🔧 Index dimension: {stats['dimension']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error caching embeddings to Pinecone: {e}")
            return False
    
    def load_cached_embeddings(
        self, 
        current_questions_metadata: Dict[int, Dict[str, Any]]
    ) -> Optional[tuple]:
        """
        Load precomputed embeddings and keywords from Pinecone
        
        Args:
            current_questions_metadata: Current questions metadata for validation
            
        Returns:
            Tuple of (key_point_embeddings, key_point_keywords) if successful, None otherwise
        """
        try:
            # Check if index has any vectors
            stats = self._index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                print("⚠️ No vectors found in Pinecone index")
                return None
            
            # Validate questions consistency
            if not self._validate_questions_consistency(current_questions_metadata):
                print("⚠️ Questions have changed since cache was created, need to recompute")
                return None
            
            # Fetch all vectors for the current questions
            key_point_embeddings = {}
            key_point_keywords = {}
            
            for question_id in current_questions_metadata.keys():
                # Query vectors for this question
                # Get the actual index dimension from stats first
                stats = self._index.describe_index_stats()
                index_dimension = stats['dimension']
                
                query_response = self._index.query(
                    vector=[0.0] * index_dimension,  # Use actual index dimension
                    filter={"question_id": question_id},
                    top_k=10000,  # Large number to get all key points for the question
                    include_values=True,
                    include_metadata=True
                )
                
                if query_response['matches']:
                    question_embeddings = []
                    question_keywords = []
                    
                    # Sort by key_point_index to maintain order
                    matches = sorted(query_response['matches'], 
                                   key=lambda x: x['metadata']['key_point_index'])
                    
                    for match in matches:
                        question_embeddings.append(match['values'])
                        keywords_list = match['metadata'].get('keywords', [])
                        question_keywords.append(set(keywords_list))
                    
                    key_point_embeddings[question_id] = question_embeddings
                    key_point_keywords[question_id] = question_keywords
            
            if key_point_embeddings:
                print(f"✅ Loaded embeddings from Pinecone")
                print(f"   📊 {len(key_point_embeddings)} questions loaded")
                print(f"   🤖 Model: {settings.openai.model_name}")
                
                return key_point_embeddings, key_point_keywords
            else:
                print("⚠️ No matching embeddings found in Pinecone")
                return None
            
        except Exception as e:
            print(f"❌ Error loading embeddings from Pinecone: {e}")
            return None
    
    def _validate_questions_consistency(
        self, 
        current_metadata: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Check if current questions match the cached questions in Pinecone
        
        Args:
            current_metadata: Current questions metadata
            
        Returns:
            True if questions are consistent, False otherwise
        """
        try:
            # Sample a few questions to check consistency
            sample_question_ids = list(current_metadata.keys())[:5]  # Check first 5 questions
            
            # Get the actual index dimension
            stats = self._index.describe_index_stats()
            index_dimension = stats['dimension']
            
            for question_id in sample_question_ids:
                # Query for this question's vectors
                query_response = self._index.query(
                    vector=[0.0] * index_dimension,
                    filter={"question_id": question_id},
                    top_k=1,
                    include_metadata=True
                )
                
                if query_response['matches']:
                    cached_metadata = query_response['matches'][0]['metadata']
                    current_meta = current_metadata[question_id]
                    
                    # Check if question text and key points count match
                    if (cached_metadata.get("question_text") != current_meta.get("question_text") or
                        cached_metadata.get("key_points_count") != current_meta.get("key_points_count")):
                        print(f"❌ Question {question_id} content changed")
                        return False
                else:
                    print(f"❌ Question {question_id} not found in cache")
                    return False
            
            return True
            
        except Exception as e:
            print(f"❌ Error validating questions consistency: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Delete all vectors from the Pinecone index
        
        Returns:
            True if cleared successfully, False on error
        """
        try:
            # Delete all vectors in the index
            self._index.delete(delete_all=True)
            print(f"✅ Cleared all vectors from Pinecone index: {self._pinecone_config.index_name}")
            return True
        except Exception as e:
            print(f"❌ Error clearing Pinecone index: {e}")
            return False
    
    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current Pinecone index
        
        Returns:
            Index metadata if successful, None otherwise
        """
        try:
            stats = self._index.describe_index_stats()
            
            # Get sample metadata from one vector
            sample_metadata = {}
            if stats['total_vector_count'] > 0:
                # Use a simple query without vector to get metadata
                query_response = self._index.query(
                    vector=[0.0] * stats['dimension'],  # Use actual index dimension
                    top_k=1,
                    include_metadata=True
                )
                if query_response['matches']:
                    sample_metadata = query_response['matches'][0]['metadata']
            
            return {
                "total_vector_count": stats['total_vector_count'],
                "dimension": stats['dimension'],
                "index_name": self._pinecone_config.index_name,
                "created_at": sample_metadata.get('created_at', 'Unknown'),
                "openai_model": sample_metadata.get('openai_model', 'Unknown')
            }
            
        except Exception as e:
            print(f"❌ Error getting cache info: {e}")
            return None
