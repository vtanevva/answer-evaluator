"""
Answer cache service for storing and retrieving graded answers from Pinecone.

This service enables fast-path grading by caching student answers with their
computed scores. When a new student answer is submitted, the system checks
if a similar answer has already been graded and returns the cached score
if similarity is above the configured threshold.

Benefits:
- Instant grading for repeated/similar answers
- Reduced API calls to embedding and LLM services
- Insights into common student misconceptions
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from core.config import settings
from services.embedding_service import EmbeddingService


@dataclass
class CachedAnswerRecord:
    """
    Represents a cached answer record with metadata for Pinecone storage.
    
    Fields:
        student_answer: The original student answer text
        question_id: ID of the question being answered
        score: The computed score (0-100)
        hit_key_points: List of key points the answer covers
        missing_key_points: List of key points the answer misses
        feedback: The feedback message provided to the student
        graded_at: ISO timestamp when the answer was graded
    """
    student_answer: str
    question_id: int
    score: float
    hit_key_points: List[str]
    missing_key_points: List[str]
    feedback: str
    graded_at: str


class AnswerCacheService:
    """
    Service for storing and retrieving cached graded answers from Pinecone.
    
    This service maintains a separate namespace in Pinecone for answer caching,
    allowing fast retrieval of previously graded answers when similarity
    is above the configured threshold.
    """
    
    def __init__(self, embedding_service: EmbeddingService, pinecone_index):
        """
        Initialize the answer cache service.
        
        Args:
            embedding_service: EmbeddingService instance for generating embeddings
            pinecone_index: Pinecone index object for storing/retrieving answers
        """
        self._embedding_service = embedding_service
        self._index = pinecone_index
        self._cache_enabled = getattr(settings.grading, 'answer_cache_enabled', True)
        self._similarity_threshold = getattr(
            settings.grading, 'answer_cache_similarity_threshold', 0.99
        )
        self._max_results = getattr(
            settings.grading, 'answer_cache_max_results', 5
        )
        
        print(f"🗂️  Answer cache service initialized")
        print(f"   Enabled: {self._cache_enabled}")
        print(f"   Similarity threshold: {self._similarity_threshold}")
        print(f"   Max results: {self._max_results}")
    
    def is_enabled(self) -> bool:
        """Check if answer caching is enabled in configuration."""
        return self._cache_enabled
    
    def retrieve_similar_cached_answers(
        self, 
        question_id: int, 
        student_answer: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Retrieve similar cached answers for a given question.
        
        If a cached answer with similarity ≥ threshold is found, returns the
        cached grade without recomputation.
        
        Args:
            question_id: ID of the question
            student_answer: The student's answer text
            
        Returns:
            Tuple of (cached_answer_embedding_id, cached_record_dict) if found with high similarity,
            None otherwise
        """
        if not self._cache_enabled:
            return None
        
        try:
            # Generate embedding for student answer
            answer_embedding = self._embedding_service.get_embedding(student_answer)
            
            # Query Pinecone for similar answers to this question
            stats = self._index.describe_index_stats()
            index_dimension = stats['dimension']
            
            query_response = self._index.query(
                vector=answer_embedding,
                filter={"answer_type": "student_answer", "question_id": question_id},
                top_k=self._max_results,
                include_values=False,
                include_metadata=True
            )
            
            if not query_response['matches']:
                return None
            
            # Check if best match meets similarity threshold
            best_match = query_response['matches'][0]
            similarity_score = best_match['score']
            
            if similarity_score >= self._similarity_threshold:
                cached_record = best_match['metadata']
                cached_id = best_match['id']
                
                print(f"\n✅ HIGH SIMILARITY CACHED ANSWER FOUND")
                print(f"   Question ID: {question_id}")
                print(f"   Similarity: {similarity_score:.4f} (threshold: {self._similarity_threshold})")
                print(f"   Cached score: {cached_record.get('score', 'N/A')}")
                print(f"   Using cached grade instead of recomputing")
                
                return cached_id, cached_record
            else:
                print(f"\n📊 Similar cached answers found but below threshold:")
                for i, match in enumerate(query_response['matches'][:3]):
                    sim = match['score']
                    score = match['metadata'].get('score', 'N/A')
                    print(f"   {i+1}. Similarity: {sim:.4f}, Score: {score}")
                
                return None
            
        except Exception as e:
            print(f"⚠️  Error retrieving cached answers: {e}")
            return None
    
    def cache_graded_answer(
        self,
        question_id: int,
        student_answer: str,
        score: float,
        hit_key_points: List[str],
        missing_key_points: List[str],
        feedback: str
    ) -> bool:
        """
        Cache a newly graded answer to Pinecone for future use.
        
        Args:
            question_id: ID of the question
            student_answer: The student's answer text
            score: The computed score (0-100)
            hit_key_points: List of key points covered by the answer
            missing_key_points: List of key points missed by the answer
            feedback: Feedback message for the student
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self._cache_enabled:
            return False
        
        try:
            # Generate embedding for the answer
            answer_embedding = self._embedding_service.get_embedding(student_answer)
            
            # Create unique ID for this cached answer
            # Use question_id + timestamp to ensure uniqueness
            timestamp = int(time.time() * 1000)
            vector_id = f"answer_q{question_id}_{timestamp}"
            
            # Prepare metadata record
            metadata = {
                "answer_type": "student_answer",
                "question_id": question_id,
                "student_answer": student_answer,
                "score": score,
                "hit_key_points": hit_key_points,
                "missing_key_points": missing_key_points,
                "feedback": feedback,
                "graded_at": datetime.now().isoformat(),
                "embedding_model": settings.embeddings.model,
                "embedding_type": settings.embeddings.type
            }
            
            # Upsert to Pinecone
            self._index.upsert(vectors=[{
                "id": vector_id,
                "values": answer_embedding,
                "metadata": metadata
            }])
            
            print(f"\n💾 CACHED ANSWER SAVED")
            print(f"   Question ID: {question_id}")
            print(f"   Vector ID: {vector_id}")
            print(f"   Score: {score}")
            print(f"   Cached for future similar submissions")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Error caching answer: {e}")
            return False
    
    def get_cache_stats(self, question_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get statistics about cached answers, optionally filtered by question_id.
        
        Args:
            question_id: Optional question ID to filter cache stats
            
        Returns:
            Dictionary with cache statistics or None on error
        """
        try:
            if question_id is not None:
                # Get stats for specific question
                stats = self._index.describe_index_stats()
                index_dimension = stats['dimension']
                
                query_response = self._index.query(
                    vector=[0.0] * index_dimension,
                    filter={"answer_type": "student_answer", "question_id": question_id},
                    top_k=10000,
                    include_metadata=True
                )
                
                cached_count = len(query_response['matches'])
                avg_score = 0.0
                
                if cached_count > 0:
                    scores = [m['metadata'].get('score', 0) for m in query_response['matches']]
                    avg_score = sum(scores) / len(scores)
                
                return {
                    "question_id": question_id,
                    "cached_answers_count": cached_count,
                    "average_score": round(avg_score, 2)
                }
            else:
                # Get overall stats
                stats = self._index.describe_index_stats()
                return {
                    "total_vectors": stats.get('total_vector_count', 0),
                    "dimension": stats.get('dimension', 0),
                    "index_name": stats.get('index_name', 'unknown')
                }
        
        except Exception as e:
            print(f"⚠️  Error getting cache stats: {e}")
            return None
    
    def clear_cache_for_question(self, question_id: int) -> bool:
        """
        Clear all cached answers for a specific question.
        
        Useful when question content is updated or corrections are needed.
        
        Args:
            question_id: ID of the question to clear
            
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            self._index.delete(filter={"question_id": question_id, "answer_type": "student_answer"})
            print(f"✅ Cleared cache for question {question_id}")
            return True
        except Exception as e:
            print(f"⚠️  Error clearing cache: {e}")
            return False
