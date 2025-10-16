"""
AI-powered antonym detection service using embeddings and semantic analysis
"""

import numpy as np
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
    context_similarity: float


class AIAntonymDetector:
    """
    AI-powered antonym detection using multiple approaches:
    1. Embedding-based semantic distance analysis
    2. Context-aware sentence-level analysis
    3. LLM-based validation for complex cases
    4. Hybrid approach combining multiple signals
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client=None):
        """
        Initialize AI antonym detector
        
        Args:
            embedding_service: Service for generating embeddings
            openai_client: OpenAI client for LLM-based validation (optional)
        """
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        
        # Configuration
        self._config = getattr(settings, 'antonym_detection', {
            'semantic_distance_threshold': 0.3,  # Lower = more similar (antonyms should be far apart)
            'context_similarity_threshold': 0.7,  # High context similarity + low semantic similarity = antonym
            'confidence_thresholds': {
                'high': 0.8,
                'medium': 0.6,
                'low': 0.4
            },
            'use_llm_validation': True,
            'llm_validation_threshold': 0.5  # Use LLM when confidence is below this
        })
        
        # Known antonym patterns for validation
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
            ("agree", "disagree"), ("accept", "reject"), ("approve", "disapprove")
        }
        
        # Negation prefixes for enhanced detection
        self._negation_prefixes = {
            "un", "dis", "in", "im", "ir", "il", "non", "de", "anti", 
            "counter", "mis", "mal", "pseudo", "quasi", "semi"
        }
        
        # Contextual indicators that suggest antonym relationships
        self._antonym_indicators = {
            "opposite", "contrary", "reverse", "inverse", "contrast",
            "versus", "vs", "against", "instead", "rather", "but", "however",
            "although", "despite", "whereas", "while", "on the other hand"
        }
    
    def detect_antonyms(self, text1: str, text2: str, context: str = "") -> AntonymDetectionResult:
        """
        Detect if two text segments are antonyms using AI-powered analysis
        
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
                context_similarity=1.0
            )
        
        # Method 1: Semantic distance analysis using embeddings
        semantic_result = self._analyze_semantic_distance(text1_clean, text2_clean)
        
        # Method 2: Context-aware analysis
        context_result = self._analyze_contextual_relationship(text1_clean, text2_clean, context)
        
        # Method 3: Pattern-based validation
        pattern_result = self._validate_known_patterns(text1_clean, text2_clean)
        
        # Method 4: LLM-based validation (if available and needed)
        llm_result = None
        if (self._openai_client and 
            self._config.get('use_llm_validation', True) and
            semantic_result.confidence == AntonymConfidence.MEDIUM):
            llm_result = self._llm_validate_antonyms(text1_clean, text2_clean, context)
        
        # Combine results using weighted scoring
        final_result = self._combine_detection_results(
            semantic_result, context_result, pattern_result, llm_result
        )
        
        return final_result
    
    def _analyze_semantic_distance(self, text1: str, text2: str) -> AntonymDetectionResult:
        """
        Analyze semantic distance between two texts using embeddings
        """
        try:
            # Get embeddings for both texts
            embedding1 = self._embedding_service.get_embedding(text1)
            embedding2 = self._embedding_service.get_embedding(text2)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(embedding1, embedding2)
            semantic_distance = 1.0 - similarity
            
            # Determine if this suggests antonym relationship
            # Antonyms should have low similarity (high distance) but not too low (unrelated)
            is_antonym = (semantic_distance > self._config['semantic_distance_threshold'] and 
                         similarity > 0.1)  # Not completely unrelated
            
            # Calculate confidence based on distance
            if semantic_distance > 0.8:
                confidence = AntonymConfidence.HIGH
            elif semantic_distance > 0.6:
                confidence = AntonymConfidence.MEDIUM
            elif semantic_distance > 0.4:
                confidence = AntonymConfidence.LOW
            else:
                confidence = AntonymConfidence.NONE
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="semantic_distance",
                evidence=f"Semantic distance: {semantic_distance:.3f}",
                semantic_distance=semantic_distance,
                context_similarity=0.0  # Not applicable for this method
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="semantic_distance",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0
            )
    
    def _analyze_contextual_relationship(self, text1: str, text2: str, context: str) -> AntonymDetectionResult:
        """
        Analyze contextual relationship between texts
        """
        # Check for antonym indicators in context
        context_lower = context.lower()
        has_antonym_indicators = any(indicator in context_lower for indicator in self._antonym_indicators)
        
        # Check for negation patterns
        has_negation = self._detect_negation_pattern(text1, text2)
        
        # Calculate context similarity
        context_similarity = 0.0
        if context:
            try:
                context_embedding = self._embedding_service.get_embedding(context)
                text1_embedding = self._embedding_service.get_embedding(text1)
                text2_embedding = self._embedding_service.get_embedding(text2)
                
                # Average similarity of both texts to context
                sim1 = self._cosine_similarity(context_embedding, text1_embedding)
                sim2 = self._cosine_similarity(context_embedding, text2_embedding)
                context_similarity = (sim1 + sim2) / 2
            except:
                pass
        
        # Determine antonym relationship based on context
        is_antonym = has_antonym_indicators or has_negation
        
        # Calculate confidence
        confidence_score = 0.0
        if has_antonym_indicators:
            confidence_score += 0.6
        if has_negation:
            confidence_score += 0.4
        if context_similarity > 0.7:  # High context similarity supports antonym relationship
            confidence_score += 0.2
        
        if confidence_score >= 0.8:
            confidence = AntonymConfidence.HIGH
        elif confidence_score >= 0.6:
            confidence = AntonymConfidence.MEDIUM
        elif confidence_score >= 0.4:
            confidence = AntonymConfidence.LOW
        else:
            confidence = AntonymConfidence.NONE
        
        evidence_parts = []
        if has_antonym_indicators:
            evidence_parts.append("antonym indicators found")
        if has_negation:
            evidence_parts.append("negation pattern detected")
        if context_similarity > 0.7:
            evidence_parts.append(f"high context similarity: {context_similarity:.3f}")
        
        return AntonymDetectionResult(
            is_antonym=is_antonym,
            confidence=confidence,
            method="contextual_analysis",
            evidence="; ".join(evidence_parts) if evidence_parts else "no contextual evidence",
            semantic_distance=0.0,  # Not calculated in this method
            context_similarity=context_similarity
        )
    
    def _validate_known_patterns(self, text1: str, text2: str) -> AntonymDetectionResult:
        """
        Validate against known antonym patterns
        """
        # Check exact matches in known patterns
        for pattern in self._known_antonym_patterns:
            if ((text1 in pattern and text2 in pattern) or
                (text2 in pattern and text1 in pattern)):
                return AntonymDetectionResult(
                    is_antonym=True,
                    confidence=AntonymConfidence.HIGH,
                    method="known_pattern",
                    evidence=f"Known antonym pair: {pattern}",
                    semantic_distance=0.0,
                    context_similarity=0.0
                )
        
        # Check for negation patterns
        negation_result = self._detect_negation_pattern(text1, text2)
        if negation_result:
            return AntonymDetectionResult(
                is_antonym=True,
                confidence=AntonymConfidence.MEDIUM,
                method="negation_pattern",
                evidence="Negation pattern detected",
                semantic_distance=0.0,
                context_similarity=0.0
            )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="known_pattern",
            evidence="No known patterns matched",
            semantic_distance=0.0,
            context_similarity=0.0
        )
    
    def _detect_negation_pattern(self, text1: str, text2: str) -> bool:
        """
        Detect if texts differ by negation prefixes
        """
        # Check if one text is the negation of the other
        for prefix in self._negation_prefixes:
            if text1.startswith(prefix) and text1[len(prefix):] == text2:
                return True
            if text2.startswith(prefix) and text2[len(prefix):] == text1:
                return True
        
        return False
    
    def _llm_validate_antonyms(self, text1: str, text2: str, context: str) -> Optional[AntonymDetectionResult]:
        """
        Use LLM to validate antonym relationship for complex cases
        """
        if not self._openai_client:
            return None
        
        try:
            prompt = f"""
            Analyze if the following two phrases are antonyms (opposites) in meaning:

            Phrase 1: "{text1}"
            Phrase 2: "{text2}"
            Context: "{context}"

            Consider:
            - Are they direct opposites?
            - Do they represent contrasting concepts?
            - Are they mutually exclusive in the given context?

            Respond with:
            - "YES" if they are antonyms
            - "NO" if they are not antonyms
            - "UNCERTAIN" if unclear

            Also provide a brief explanation.
            """
            
            response = self._openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip().upper()
            
            is_antonym = "YES" in result_text
            confidence = AntonymConfidence.HIGH if "YES" in result_text else AntonymConfidence.LOW
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="llm_validation",
                evidence=f"LLM analysis: {result_text}",
                semantic_distance=0.0,
                context_similarity=0.0
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="llm_validation",
                evidence=f"LLM error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0
            )
    
    def _combine_detection_results(self, *results: AntonymDetectionResult) -> AntonymDetectionResult:
        """
        Combine multiple detection results using weighted scoring
        """
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="combined",
                evidence="No valid results",
                semantic_distance=0.0,
                context_similarity=0.0
            )
        
        # Weight different methods
        weights = {
            "semantic_distance": 0.4,
            "contextual_analysis": 0.3,
            "known_pattern": 0.2,
            "negation_pattern": 0.2,
            "llm_validation": 0.3
        }
        
        # Calculate weighted scores
        total_weight = 0.0
        weighted_antonym_score = 0.0
        weighted_confidence_score = 0.0
        
        evidence_parts = []
        semantic_distances = []
        context_similarities = []
        
        for result in valid_results:
            weight = weights.get(result.method, 0.1)
            total_weight += weight
            
            if result.is_antonym:
                weighted_antonym_score += weight
            
            # Convert confidence to numeric score
            confidence_scores = {
                AntonymConfidence.HIGH: 1.0,
                AntonymConfidence.MEDIUM: 0.7,
                AntonymConfidence.LOW: 0.4,
                AntonymConfidence.NONE: 0.0
            }
            weighted_confidence_score += weight * confidence_scores[result.confidence]
            
            evidence_parts.append(f"{result.method}: {result.evidence}")
            if result.semantic_distance > 0:
                semantic_distances.append(result.semantic_distance)
            if result.context_similarity > 0:
                context_similarities.append(result.context_similarity)
        
        # Normalize scores
        if total_weight > 0:
            final_antonym_score = weighted_antonym_score / total_weight
            final_confidence_score = weighted_confidence_score / total_weight
        else:
            final_antonym_score = 0.0
            final_confidence_score = 0.0
        
        # Determine final result
        is_antonym = final_antonym_score > 0.5
        
        if final_confidence_score >= 0.8:
            confidence = AntonymConfidence.HIGH
        elif final_confidence_score >= 0.6:
            confidence = AntonymConfidence.MEDIUM
        elif final_confidence_score >= 0.4:
            confidence = AntonymConfidence.LOW
        else:
            confidence = AntonymConfidence.NONE
        
        # Calculate average semantic distance and context similarity
        avg_semantic_distance = np.mean(semantic_distances) if semantic_distances else 0.0
        avg_context_similarity = np.mean(context_similarities) if context_similarities else 0.0
        
        return AntonymDetectionResult(
            is_antonym=is_antonym,
            confidence=confidence,
            method="combined",
            evidence="; ".join(evidence_parts),
            semantic_distance=avg_semantic_distance,
            context_similarity=avg_context_similarity
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def batch_detect_antonyms(self, text_pairs: List[Tuple[str, str]], 
                            contexts: List[str] = None) -> List[AntonymDetectionResult]:
        """
        Detect antonyms for multiple text pairs efficiently
        
        Args:
            text_pairs: List of (text1, text2) tuples
            contexts: Optional list of contexts for each pair
            
        Returns:
            List of AntonymDetectionResult objects
        """
        if contexts is None:
            contexts = [""] * len(text_pairs)
        
        results = []
        for i, (text1, text2) in enumerate(text_pairs):
            context = contexts[i] if i < len(contexts) else ""
            result = self.detect_antonyms(text1, text2, context)
            results.append(result)
        
        return results
