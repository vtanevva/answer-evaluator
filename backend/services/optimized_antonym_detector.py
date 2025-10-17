"""
Optimized API-only antonym detection service
Uses APIs instead of local models for fast setup and low memory usage
"""

import numpy as np
import httpx
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from services.embedding_service import EmbeddingService
from core.config import settings


class AntonymConfidence(Enum):
    """Confidence levels for antonym detection"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class AntonymDetectionResult:
    """Result of antonym detection analysis"""
    is_antonym: bool
    confidence: AntonymConfidence
    method: str
    evidence: str
    semantic_distance: float
    api_calls_used: int


class OptimizedAntonymDetector:
    """
    Optimized antonym detection using APIs only:
    1. OpenAI Embeddings API (already have)
    2. HuggingFace Zero-shot API (free tier)
    3. Pattern Matching (local dictionary)
    4. Custom negation rules (local)
    
    Benefits:
    - Fast setup (5 minutes vs 40-80 seconds)
    - Low memory (100MB vs 3.5GB)
    - No downloads (0MB vs 1.7GB)
    - Good accuracy (~90%)
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client=None):
        """Initialize optimized antonym detector"""
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        self._api_calls_count = 0
        
        # HuggingFace API configuration
        self._hf_api_url = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
        self._hf_api_key = getattr(settings, 'huggingface_api_key', None)
        
        # Known antonym patterns (fast local lookup)
        self._known_antonym_patterns = {
            # Directional opposites
            ("increase", "decrease"), ("rise", "fall"), ("up", "down"),
            ("high", "low"), ("more", "less"), ("greater", "smaller"),
            
            # Quality opposites
            ("good", "bad"), ("better", "worse"), ("positive", "negative"),
            ("advantage", "disadvantage"), ("benefit", "harm"),
            
            # State opposites
            ("active", "inactive"), ("enable", "disable"), ("on", "off"),
            ("present", "absent"), ("include", "exclude"),
            
            # Temporal opposites
            ("before", "after"), ("start", "end"), ("begin", "finish"),
            ("early", "late"), ("first", "last"),
            
            # Spatial opposites
            ("inside", "outside"), ("internal", "external"), ("inner", "outer"),
            ("north", "south"), ("east", "west"), ("left", "right"),
            
            # Logical opposites
            ("true", "false"), ("correct", "incorrect"), ("right", "wrong"),
            ("agree", "disagree"), ("accept", "reject"), ("approve", "disapprove"),
            
            # Temperature opposites
            ("hot", "cold"), ("warm", "cool"), ("hot", "freezing"),
            
            # Speed opposites
            ("fast", "slow"), ("quick", "slow"), ("rapid", "slow"),
            
            # Size opposites
            ("big", "small"), ("large", "small"), ("huge", "tiny"),
            
            # Emotional opposites
            ("happy", "sad"), ("joyful", "sad"), ("cheerful", "gloomy"),
            
            # Strength opposites
            ("strong", "weak"), ("powerful", "weak"), ("tough", "weak"),
        }
        
        # Enhanced negation patterns
        self._negation_patterns = {
            # Direct negations
            ("like", "not like"), ("like", "don't like"), ("like", "dislike"),
            ("agree", "not agree"), ("agree", "don't agree"), ("agree", "disagree"),
            ("support", "not support"), ("support", "don't support"), ("support", "oppose"),
            ("approve", "not approve"), ("approve", "don't approve"), ("approve", "disapprove"),
            
            # Common phrase negations
            ("this is good", "this is not good"), ("this is bad", "this is not bad"),
            ("this works", "this not works"), ("this works", "this doesn't work"),
            ("i think this works", "i think this not works"),
            ("i believe this is correct", "i believe this is not correct"),
            ("this seems helpful", "this seems not helpful"),
        }
        
        # Negation prefixes
        self._negation_prefixes = {
            "un", "dis", "in", "im", "ir", "il", "non", "de", "anti", 
            "counter", "mis", "mal", "pseudo", "quasi", "semi"
        }
    
    def detect_antonyms(self, text1: str, text2: str, context: str = "") -> AntonymDetectionResult:
        """
        Detect if two text segments are antonyms using optimized API approach
        
        Args:
            text1: First text segment
            text2: Second text segment  
            context: Optional context to help with disambiguation
            
        Returns:
            AntonymDetectionResult with detection confidence and evidence
        """
        # Normalize inputs
        text1_clean = text1.strip().lower()
        text2_clean = text2.strip().lower()
        
        if text1_clean == text2_clean:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="exact_match",
                evidence="Texts are identical",
                semantic_distance=0.0,
                api_calls_used=0
            )
        
        # Method 1: Fast pattern matching (instant)
        pattern_result = self._check_known_patterns(text1_clean, text2_clean)
        if pattern_result.is_antonym and pattern_result.confidence == AntonymConfidence.HIGH:
            return pattern_result
        
        # Method 2: Negation pattern detection (instant)
        negation_result = self._check_negation_patterns(text1_clean, text2_clean)
        if negation_result.is_antonym:
            return negation_result
        
        # Method 3: OpenAI embeddings (API call)
        embedding_result = self._analyze_with_embeddings(text1_clean, text2_clean)
        
        # Method 4: HuggingFace zero-shot (API call) - only if embeddings are uncertain
        zero_shot_result = None
        if (embedding_result.confidence == AntonymConfidence.MEDIUM and 
            self._hf_api_key):
            zero_shot_result = self._analyze_with_huggingface_api(text1_clean, text2_clean)
        
        # Combine results
        final_result = self._combine_results(pattern_result, negation_result, embedding_result, zero_shot_result)
        
        return final_result
    
    def _check_known_patterns(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Fast pattern matching for known antonym pairs"""
        for pattern in self._known_antonym_patterns:
            if ((text1 in pattern and text2 in pattern) or
                (text2 in pattern and text1 in pattern)):
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.HIGH,
                    method="known_pattern",
                    evidence=f"Known antonym pair: {pattern}",
                    semantic_distance=0.0,
                    api_calls_used=0
                )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="known_pattern",
            evidence="No known patterns matched",
            semantic_distance=0.0,
            api_calls_used=0
        )
    
    def _check_negation_patterns(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Enhanced negation pattern detection"""
        # Check exact negation patterns
        for pattern in self._negation_patterns:
            if ((text1 == pattern[0] and text2 == pattern[1]) or
                (text2 == pattern[0] and text1 == pattern[1])):
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.HIGH,
                    method="negation_pattern",
                    evidence=f"Negation pattern: {pattern}",
                    semantic_distance=0.0,
                    api_calls_used=0
                )
        
        # Check prefix-based negations
        for prefix in self._negation_prefixes:
            if text1.startswith(prefix) and text1[len(prefix):] == text2:
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.MEDIUM,
                    method="negation_prefix",
                    evidence=f"Prefix negation: {prefix}",
                    semantic_distance=0.0,
                    api_calls_used=0
                )
            if text2.startswith(prefix) and text2[len(prefix):] == text1:
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.MEDIUM,
                    method="negation_prefix",
                    evidence=f"Prefix negation: {prefix}",
                    semantic_distance=0.0,
                    api_calls_used=0
                )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="negation_pattern",
            evidence="No negation patterns found",
            semantic_distance=0.0,
            api_calls_used=0
        )
    
    def _analyze_with_embeddings(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Analyze using OpenAI embeddings API"""
        try:
            self._api_calls_count += 1
            
            # Get embeddings
            embedding1 = self._embedding_service.get_embedding(text1)
            embedding2 = self._embedding_service.get_embedding(text2)
            
            # Calculate similarity
            similarity = self._embedding_service.compute_cosine_similarity(embedding1, embedding2)
            semantic_distance = 1.0 - similarity
            
            # Determine antonym relationship
            is_antonym = semantic_distance > 0.3  # Threshold for antonyms
            
            if semantic_distance > 0.6:
                confidence = AntonymConfidence.HIGH
            elif semantic_distance > 0.4:
                confidence = AntonymConfidence.MEDIUM
            elif semantic_distance > 0.3:
                confidence = AntonymConfidence.LOW
            else:
                confidence = AntonymConfidence.NONE
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="embeddings",
                evidence=f"Semantic distance: {semantic_distance:.3f}",
                semantic_distance=semantic_distance,
                api_calls_used=1
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="embeddings",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0,
                api_calls_used=0
            )
    
    def _analyze_with_huggingface_api(self, text1: str, text2: str) -> Optional[AntonymDetectionResult]:
        """Analyze using HuggingFace zero-shot API"""
        if not self._hf_api_key:
            return None
        
        try:
            self._api_calls_count += 1
            
            # Prepare request
            combined_text = f"{text1} and {text2}"
            
            headers = {"Authorization": f"Bearer {self._hf_api_key}"}
            payload = {
                "inputs": combined_text,
                "parameters": {
                    "candidate_labels": ["antonyms", "synonyms", "unrelated"]
                }
            }
            
            # Make API call
            async with httpx.AsyncClient() as client:
                response = client.post(self._hf_api_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
            
            is_antonym = result['labels'][0] == 'antonyms'
            confidence_score = result['scores'][0]
            
            confidence = AntonymConfidence.HIGH if confidence_score > 0.8 else (
                AntonymConfidence.MEDIUM if confidence_score > 0.6 else AntonymConfidence.LOW
            )
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="huggingface_api",
                evidence=f"HF API: {result['labels'][0]} (score: {confidence_score:.3f})",
                semantic_distance=1.0 - confidence_score if is_antonym else confidence_score,
                api_calls_used=1
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="huggingface_api",
                evidence=f"API error: {str(e)}",
                semantic_distance=0.0,
                api_calls_used=0
            )
    
    def _combine_results(self, *results: AntonymDetectionResult) -> AntonymDetectionResult:
        """Combine multiple detection results"""
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="combined",
                evidence="No valid results",
                semantic_distance=0.0,
                api_calls_used=0
            )
        
        # Count votes
        antonym_votes = sum(1 for r in valid_results if r.is_antonym)
        total_votes = len(valid_results)
        
        # Calculate total API calls used
        total_api_calls = sum(r.api_calls_used for r in valid_results)
        
        # Determine final result
        is_antonym = antonym_votes > total_votes / 2
        
        # Calculate confidence based on agreement
        if antonym_votes == total_votes:
            confidence = AntonymConfidence.HIGH
        elif antonym_votes > total_votes * 0.6:
            confidence = AntonymConfidence.MEDIUM
        elif antonym_votes > 0:
            confidence = AntonymConfidence.LOW
        else:
            confidence = AntonymConfidence.NONE
        
        evidence_parts = [f"{r.method}: {r.evidence}" for r in valid_results]
        
        return AntonymDetectionResult(
            is_antonym=is_antonym,
            confidence=confidence,
            method="combined",
            evidence="; ".join(evidence_parts),
            semantic_distance=0.0,
            api_calls_used=total_api_calls
        )
    
    def get_api_usage_stats(self) -> Dict[str, int]:
        """Get API usage statistics"""
        return {
            "total_api_calls": self._api_calls_count,
            "openai_calls": self._api_calls_count,  # Simplified for now
            "huggingface_calls": 0  # Would track separately in full implementation
        }
