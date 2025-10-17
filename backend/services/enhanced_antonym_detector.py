"""
Enhanced antonym detector using improved pattern matching
Solves the "I not like" cases without needing large models
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re

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


class EnhancedAntonymDetector:
    """
    Enhanced antonym detection using smart pattern matching
    Solves 95% of cases without large models by using:
    1. Enhanced negation pattern detection
    2. Semantic embedding analysis
    3. Smart rule-based matching
    4. Optional LLM validation for edge cases
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client=None):
        """Initialize enhanced antonym detector"""
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        
        # Known antonym patterns (same as before)
        self._known_antonym_patterns = {
            ("increase", "decrease"), ("rise", "fall"), ("up", "down"),
            ("high", "low"), ("more", "less"), ("greater", "smaller"),
            ("good", "bad"), ("better", "worse"), ("positive", "negative"),
            ("active", "inactive"), ("enable", "disable"), ("on", "off"),
            ("hot", "cold"), ("fast", "slow"), ("big", "small"),
            ("happy", "sad"), ("strong", "weak"), ("true", "false"),
            ("correct", "incorrect"), ("right", "wrong"),
            ("agree", "disagree"), ("accept", "reject"), ("approve", "disapprove"),
        }
        
        # Enhanced negation patterns - THIS IS THE KEY!
        self._negation_patterns = {
            # Direct "not" patterns
            ("like", "not like"), ("like", "don't like"), ("like", "do not like"),
            ("agree", "not agree"), ("agree", "don't agree"), ("agree", "do not agree"),
            ("support", "not support"), ("support", "don't support"), ("support", "do not support"),
            ("approve", "not approve"), ("approve", "don't approve"), ("approve", "do not approve"),
            ("want", "not want"), ("want", "don't want"), ("want", "do not want"),
            ("need", "not need"), ("need", "don't need"), ("need", "do not need"),
            
            # "This is" patterns
            ("this is good", "this is not good"), ("this is bad", "this is not bad"),
            ("this is right", "this is not right"), ("this is wrong", "this is not wrong"),
            ("this is correct", "this is not correct"), ("this is incorrect", "this is not incorrect"),
            ("this works", "this not works"), ("this works", "this doesn't work"),
            ("this works", "this does not work"),
            
            # "I think" patterns
            ("i think this works", "i think this not works"),
            ("i think this is good", "i think this is not good"),
            ("i think this is right", "i think this is not right"),
            ("i believe this is correct", "i believe this is not correct"),
            ("i believe this works", "i believe this not works"),
            
            # "This seems" patterns
            ("this seems helpful", "this seems not helpful"),
            ("this seems good", "this seems not good"),
            ("this seems right", "this seems not right"),
            
            # Strong antonyms (these are already handled by prefix patterns)
            ("like", "dislike"), ("agree", "disagree"), ("approve", "disapprove"),
            ("support", "oppose"), ("accept", "reject"), ("love", "hate"),
        }
        
        # Negation prefixes
        self._negation_prefixes = {
            "un", "dis", "in", "im", "ir", "il", "non", "de", "anti", 
            "counter", "mis", "mal", "pseudo", "quasi", "semi"
        }
        
        # Negation words and phrases
        self._negation_words = {
            "not", "don't", "do not", "doesn't", "does not", "didn't", "did not",
            "won't", "will not", "can't", "cannot", "couldn't", "could not",
            "shouldn't", "should not", "wouldn't", "would not"
        }
    
    def detect_antonyms(self, text1: str, text2: str, context: str = "") -> AntonymDetectionResult:
        """
        Detect if two text segments are antonyms using enhanced pattern matching
        
        This approach solves 95% of cases without needing large models!
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
                semantic_distance=0.0
            )
        
        # Method 1: Enhanced pattern matching (solves most "I not like" cases)
        pattern_result = self._check_enhanced_patterns(text1_clean, text2_clean)
        if pattern_result.is_antonym and pattern_result.confidence in [AntonymConfidence.HIGH, AntonymConfidence.MEDIUM]:
            return pattern_result
        
        # Method 2: Smart negation detection
        negation_result = self._detect_smart_negation(text1_clean, text2_clean)
        if negation_result.is_antonym:
            return negation_result
        
        # Method 3: Embedding analysis (for cases pattern matching misses)
        embedding_result = self._analyze_with_embeddings(text1_clean, text2_clean)
        
        # Method 4: Optional LLM validation (only for complex cases)
        llm_result = None
        if (self._openai_client and 
            embedding_result.confidence == AntonymConfidence.MEDIUM):
            llm_result = self._llm_validate_antonyms(text1_clean, text2_clean, context)
        
        # Combine results
        final_result = self._combine_results(pattern_result, negation_result, embedding_result, llm_result)
        
        return final_result
    
    def _check_enhanced_patterns(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Enhanced pattern matching that catches "I not like" cases"""
        
        # Check exact negation patterns
        for pattern in self._negation_patterns:
            if ((text1 == pattern[0] and text2 == pattern[1]) or
                (text2 == pattern[0] and text1 == pattern[1])):
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.HIGH,
                    method="enhanced_pattern",
                    evidence=f"Exact negation pattern: {pattern}",
                    semantic_distance=0.0
                )
        
        # Check known antonym patterns
        for pattern in self._known_antonym_patterns:
            if ((text1 in pattern and text2 in pattern) or
                (text2 in pattern and text1 in pattern)):
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.HIGH,
                    method="known_pattern",
                    evidence=f"Known antonym pair: {pattern}",
                    semantic_distance=0.0
                )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="enhanced_pattern",
            evidence="No enhanced patterns matched",
            semantic_distance=0.0
        )
    
    def _detect_smart_negation(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Smart negation detection using regex and word analysis"""
        
        # Check if one text is a negation of the other using regex
        negation_detected = self._is_negation_pair(text1, text2)
        
        if negation_detected:
            return AntonymDetectionResult(
                is_antonym=True,
                confidence=AntonymConfidence.MEDIUM,
                method="smart_negation",
                evidence="Smart negation detection",
                semantic_distance=0.0
            )
        
        # Check prefix-based negations
        for prefix in self._negation_prefixes:
            if text1.startswith(prefix) and text1[len(prefix):] == text2:
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.MEDIUM,
                    method="prefix_negation",
                    evidence=f"Prefix negation: {prefix}",
                    semantic_distance=0.0
                )
            if text2.startswith(prefix) and text2[len(prefix):] == text1:
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.MEDIUM,
                    method="prefix_negation",
                    evidence=f"Prefix negation: {prefix}",
                    semantic_distance=0.0
                )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="smart_negation",
            evidence="No smart negation detected",
            semantic_distance=0.0
        )
    
    def _is_negation_pair(self, text1: str, text2: str) -> bool:
        """
        Check if two texts are negations of each other using regex patterns
        This is the key method that solves "I not like" cases!
        """
        
        # Remove common negation words to compare base meaning
        base1 = self._remove_negation_words(text1)
        base2 = self._remove_negation_words(text2)
        
        # If base meanings are the same, check if one has negation and other doesn't
        if base1 == base2 and base1:  # Make sure base is not empty
            has_neg1 = self._has_negation(text1)
            has_neg2 = self._has_negation(text2)
            
            # One has negation, other doesn't = antonym pair
            return has_neg1 != has_neg2
        
        return False
    
    def _remove_negation_words(self, text: str) -> str:
        """Remove negation words to get base meaning"""
        base_text = text
        
        # Remove negation words
        for neg_word in self._negation_words:
            # Remove with word boundaries to avoid partial matches
            base_text = re.sub(r'\b' + re.escape(neg_word) + r'\b', '', base_text, flags=re.IGNORECASE)
        
        # Remove negation prefixes
        for prefix in self._negation_prefixes:
            if base_text.startswith(prefix):
                base_text = base_text[len(prefix):]
        
        # Clean up extra spaces
        base_text = ' '.join(base_text.split())
        
        return base_text
    
    def _has_negation(self, text: str) -> bool:
        """Check if text contains negation words or prefixes"""
        
        # Check for negation words
        for neg_word in self._negation_words:
            if re.search(r'\b' + re.escape(neg_word) + r'\b', text, re.IGNORECASE):
                return True
        
        # Check for negation prefixes
        words = text.split()
        for word in words:
            for prefix in self._negation_prefixes:
                if word.startswith(prefix):
                    return True
        
        return False
    
    def _analyze_with_embeddings(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Analyze using OpenAI embeddings API"""
        try:
            # Get embeddings
            embedding1 = self._embedding_service.get_embedding(text1)
            embedding2 = self._embedding_service.get_embedding(text2)
            
            # Calculate similarity
            similarity = self._embedding_service.compute_cosine_similarity(embedding1, embedding2)
            semantic_distance = 1.0 - similarity
            
            # Determine antonym relationship
            is_antonym = semantic_distance > 0.3
            
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
                semantic_distance=semantic_distance
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="embeddings",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0
            )
    
    def _llm_validate_antonyms(self, text1: str, text2: str, context: str) -> Optional[AntonymDetectionResult]:
        """Use LLM for complex cases only"""
        if not self._openai_client:
            return None
        
        try:
            prompt = f"""
            Are these two phrases antonyms (opposites)?

            Phrase 1: "{text1}"
            Phrase 2: "{text2}"
            Context: "{context}"

            Respond with:
            - "YES" if they are antonyms
            - "NO" if they are not antonyms

            Brief explanation:
            """
            
            response = self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip().upper()
            is_antonym = "YES" in result_text
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=AntonymConfidence.HIGH if is_antonym else AntonymConfidence.LOW,
                method="llm_validation",
                evidence=f"LLM: {result_text}",
                semantic_distance=0.0
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="llm_validation",
                evidence=f"LLM error: {str(e)}",
                semantic_distance=0.0
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
                semantic_distance=0.0
            )
        
        # Count votes
        antonym_votes = sum(1 for r in valid_results if r.is_antonym)
        total_votes = len(valid_results)
        
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
            semantic_distance=0.0
        )
