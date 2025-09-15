import json
import os
from typing import Dict, List, Set, Optional, Any
from datetime import datetime

from core.config import settings


class EmbeddingStorage:
    """
    Service for persisting and loading precomputed embeddings to/from disk
    
    This service handles:
    - Saving embeddings and keywords to JSON file
    - Loading embeddings and keywords from JSON file
    - Validating embedding cache integrity
    - Managing embedding metadata (timestamps, question hashes)
    """
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize embedding storage service
        
        Args:
            file_path: Path to embeddings cache file. If None, uses config setting.
        """
        self._file_path = file_path or os.path.join(
            os.path.dirname(__file__), 
            "..", 
            settings.evaluation.embeddings_file_path
        )
    
    def cache_embeddings(
        self, 
        key_point_embeddings: Dict[int, List[List[float]]], 
        key_point_keywords: Dict[int, List[Set[str]]],
        questions_metadata: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Save precomputed embeddings and keywords to file
        
        Args:
            key_point_embeddings: Dictionary mapping question_id to list of embeddings
            key_point_keywords: Dictionary mapping question_id to list of keyword sets
            questions_metadata: Metadata about questions for validation
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert sets to lists for JSON serialization
            serializable_keywords = {
                question_id: [list(keyword_set) for keyword_set in keyword_sets]
                for question_id, keyword_sets in key_point_keywords.items()
            }
            
            # Create cache data structure
            cache_data = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "openai_model": settings.openai.model_name,
                    "total_questions": len(key_point_embeddings),
                    "total_embeddings": sum(len(embs) for embs in key_point_embeddings.values())
                },
                "questions_metadata": questions_metadata,
                "embeddings": {
                    str(question_id): embeddings 
                    for question_id, embeddings in key_point_embeddings.items()
                },
                "keywords": {
                    str(question_id): keywords 
                    for question_id, keywords in serializable_keywords.items()
                }
            }
            
            # Ensure directory exists
            dir_path = os.path.dirname(self._file_path)
            if dir_path:  # Only create directory if there is a directory path
                os.makedirs(dir_path, exist_ok=True)
            
            # Write to file
            with open(self._file_path, 'w', encoding='utf-8') as file:
                json.dump(cache_data, file, indent=2)
            
            print(f"✅ Saved embeddings cache to {self._file_path}")
            print(f"   📊 {cache_data['metadata']['total_questions']} questions, "
                  f"{cache_data['metadata']['total_embeddings']} embeddings")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving embeddings cache: {e}")
            return False
    
    def load_cached_embeddings(
        self, 
        current_questions_metadata: Dict[int, Dict[str, Any]]
    ) -> Optional[tuple]:
        """
        Load precomputed embeddings and keywords from file
        
        Args:
            current_questions_metadata: Current questions metadata for validation
            
        Returns:
            Tuple of (key_point_embeddings, key_point_keywords) if successful, None otherwise
        """
        if not os.path.exists(self._file_path):
            print(f"⚠️ Embeddings cache file not found: {self._file_path}")
            return None
        
        try:
            with open(self._file_path, 'r', encoding='utf-8') as file:
                cache_data = json.load(file)
            
            # Validate cache structure
            if not self._validate_cache_structure(cache_data):
                print("❌ Invalid cache file structure")
                return None
            
            # Check if questions have changed
            if not self._validate_questions_consistency(
                cache_data.get("questions_metadata", {}), 
                current_questions_metadata
            ):
                print("⚠️ Questions have changed since cache was created, need to recompute")
                return None
            
            # Convert back to proper types
            key_point_embeddings = {
                int(question_id): embeddings 
                for question_id, embeddings in cache_data["embeddings"].items()
            }
            
            key_point_keywords = {
                int(question_id): [set(keywords) for keywords in keyword_lists]
                for question_id, keyword_lists in cache_data["keywords"].items()
            }
            
            metadata = cache_data["metadata"]
            print(f"✅ Loaded embeddings cache from {self._file_path}")
            print(f"   📅 Created: {metadata.get('created_at', 'Unknown')}")
            print(f"   🤖 Model: {metadata.get('openai_model', 'Unknown')}")
            print(f"   📊 {metadata.get('total_questions', 0)} questions, "
                  f"{metadata.get('total_embeddings', 0)} embeddings")
            
            return key_point_embeddings, key_point_keywords
            
        except Exception as e:
            print(f"❌ Error loading embeddings cache: {e}")
            return None
    
    def _validate_cache_structure(self, cache_data: Dict[str, Any]) -> bool:
        """
        Validate that the cache file has the expected structure
        
        Args:
            cache_data: Loaded cache data
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_keys = ["metadata", "embeddings", "keywords"]
        
        for key in required_keys:
            if key not in cache_data:
                print(f"❌ Missing required key in cache: {key}")
                return False
        
        # Check if embeddings and keywords have matching question IDs
        embedding_ids = set(cache_data["embeddings"].keys())
        keyword_ids = set(cache_data["keywords"].keys())
        
        if embedding_ids != keyword_ids:
            print("❌ Mismatch between embedding and keyword question IDs")
            return False
        
        return True
    
    def _validate_questions_consistency(
        self, 
        cached_metadata: Dict[str, Any], 
        current_metadata: Dict[int, Dict[str, Any]]
    ) -> bool:
        """
        Check if current questions match the cached questions
        
        Args:
            cached_metadata: Metadata from cache file
            current_metadata: Current questions metadata
            
        Returns:
            True if questions are consistent, False otherwise
        """
        # Convert current metadata keys to strings for comparison
        current_metadata_str = {
            str(question_id): metadata 
            for question_id, metadata in current_metadata.items()
        }
        
        # Simple check: compare question counts and question IDs
        cached_ids = set(cached_metadata.keys())
        current_ids = set(current_metadata_str.keys())
        
        if cached_ids != current_ids:
            print(f"❌ Question IDs changed: cached={cached_ids}, current={current_ids}")
            return False
        
        # Check if question texts and key points match
        for question_id in cached_ids:
            cached_q = cached_metadata[question_id]
            current_q = current_metadata_str[question_id]
            
            if (cached_q.get("question_text") != current_q.get("question_text") or
                cached_q.get("key_points_count") != current_q.get("key_points_count")):
                print(f"❌ Question {question_id} content changed")
                return False
        
        return True
    
    def clear_cache(self) -> bool:
        """
        Delete the embeddings cache file
        
        Returns:
            True if deleted successfully or file doesn't exist, False on error
        """
        try:
            if os.path.exists(self._file_path):
                os.remove(self._file_path)
                print(f"✅ Cleared embeddings cache: {self._file_path}")
            return True
        except Exception as e:
            print(f"❌ Error clearing embeddings cache: {e}")
            return False
    
    def get_cache_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current cache file
        
        Returns:
            Cache metadata if file exists, None otherwise
        """
        if not os.path.exists(self._file_path):
            return None
        
        try:
            with open(self._file_path, 'r', encoding='utf-8') as file:
                cache_data = json.load(file)
            
            return cache_data.get("metadata", {})
            
        except Exception as e:
            print(f"❌ Error reading cache info: {e}")
            return None
