"""
Advanced AI-powered antonym detection using fine-tuned sentence transformers
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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
    model_scores: Dict[str, float]


class AdvancedAntonymDetector:
    """
    Advanced AI-powered antonym detection using multiple specialized models:
    1. Fine-tuned sentence transformer for negations
    2. Specialized antonym-synonym discrimination model
    3. Zero-shot classification for antonym detection
    4. Hybrid approach combining multiple signals
    """
    
    def __init__(self, embedding_service: EmbeddingService, openai_client=None):
        """
        Initialize advanced antonym detector
        
        Args:
            embedding_service: Service for generating embeddings
            openai_client: OpenAI client for LLM-based validation (optional)
        """
        self._embedding_service = embedding_service
        self._openai_client = openai_client
        
        # Configuration
        self._config = getattr(settings, 'antonym_detection', None)
        if self._config is None:
            from core.config import AntonymDetectionConfig
            self._config = AntonymDetectionConfig()
        
        # Initialize models
        self._initialize_models()
        
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
    
    def _initialize_models(self):
        """Initialize the AI models for antonym detection"""
        try:
            # Model 1: Fine-tuned sentence transformer for negations
            print("🔄 Loading fine-tuned negation model...")
            self._negation_model = SentenceTransformer('LeoChiuu/all-MiniLM-L6-v2-negations')
            print("✅ Negation model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load negation model: {e}")
            self._negation_model = None
        
        try:
            # Model 2: Zero-shot classifier for antonym detection
            print("🔄 Loading zero-shot classifier...")
            self._zero_shot_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
            print("✅ Zero-shot classifier loaded")
        except Exception as e:
            print(f"⚠️ Failed to load zero-shot classifier: {e}")
            self._zero_shot_classifier = None
        
        # Model 3: Standard sentence transformer as fallback
        try:
            print("🔄 Loading standard sentence transformer...")
            self._standard_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Standard model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load standard model: {e}")
            self._standard_model = None
    
    def detect_antonyms(self, text1: str, text2: str, context: str = "") -> AntonymDetectionResult:
        """
        Detect if two text segments are antonyms using advanced AI models
        
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
                context_similarity=1.0,
                model_scores={}
            )
        
        # Method 1: Fine-tuned negation model
        negation_result = self._analyze_with_negation_model(text1_clean, text2_clean)
        
        # Method 2: Zero-shot classification
        zero_shot_result = self._analyze_with_zero_shot(text1_clean, text2_clean, context)
        
        # Method 3: Standard semantic analysis
        semantic_result = self._analyze_with_standard_model(text1_clean, text2_clean)
        
        # Method 4: Pattern-based validation
        pattern_result = self._validate_known_patterns(text1_clean, text2_clean)
        
        # Method 5: LLM-based validation (if available and needed)
        llm_result = None
        if (self._openai_client and 
            self._config.use_llm_validation and
            self._needs_llm_validation(negation_result, zero_shot_result, semantic_result)):
            llm_result = self._llm_validate_antonyms(text1_clean, text2_clean, context)
        
        # Combine results using advanced weighted scoring
        final_result = self._combine_detection_results(
            negation_result, zero_shot_result, semantic_result, pattern_result, llm_result
        )
        
        return final_result
    
    def _analyze_with_negation_model(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Analyze using fine-tuned negation model"""
        if not self._negation_model:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="negation_model",
                evidence="Model not available",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
        
        try:
            # Get embeddings using negation model
            embeddings = self._negation_model.encode([text1, text2])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_distance = 1.0 - similarity
            
            # For negation model, we expect antonyms to have higher distance
            is_antonym = semantic_distance > 0.3  # Higher threshold for negation model
            
            confidence = AntonymConfidence.HIGH if semantic_distance > 0.5 else (
                AntonymConfidence.MEDIUM if semantic_distance > 0.3 else AntonymConfidence.LOW
            )
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="negation_model",
                evidence=f"Negation model distance: {semantic_distance:.3f}",
                semantic_distance=semantic_distance,
                context_similarity=0.0,
                model_scores={"negation_model": semantic_distance}
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="negation_model",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
    
    def _analyze_with_zero_shot(self, text1: str, text2: str, context: str) -> AntonymDetectionResult:
        """Analyze using zero-shot classification"""
        if not self._zero_shot_classifier:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="zero_shot",
                evidence="Model not available",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
        
        try:
            # Create a combined text for classification
            combined_text = f"{text1} and {text2}"
            if context:
                combined_text = f"{context}: {combined_text}"
            
            # Classify as antonym or synonym
            result = self._zero_shot_classifier(
                combined_text,
                candidate_labels=["antonyms", "synonyms", "unrelated"],
                hypothesis_template="These words are {}.", 
                multi_label=False
            )
            
            is_antonym = result['labels'][0] == 'antonyms'
            confidence_score = result['scores'][0]
            
            confidence = AntonymConfidence.HIGH if confidence_score > 0.8 else (
                AntonymConfidence.MEDIUM if confidence_score > 0.6 else AntonymConfidence.LOW
            )
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="zero_shot",
                evidence=f"Zero-shot: {result['labels'][0]} (score: {confidence_score:.3f})",
                semantic_distance=1.0 - confidence_score if is_antonym else confidence_score,
                context_similarity=0.0,
                model_scores={"zero_shot": confidence_score}
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="zero_shot",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
    
    def _analyze_with_standard_model(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Analyze using standard sentence transformer"""
        if not self._standard_model:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="standard_model",
                evidence="Model not available",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
        
        try:
            # Get embeddings using standard model
            embeddings = self._standard_model.encode([text1, text2])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            semantic_distance = 1.0 - similarity
            
            # For standard model, antonyms might have moderate distance
            is_antonym = semantic_distance > 0.15  # Lower threshold for standard model
            
            confidence = AntonymConfidence.HIGH if semantic_distance > 0.3 else (
                AntonymConfidence.MEDIUM if semantic_distance > 0.15 else AntonymConfidence.LOW
            )
            
            return AntonymDetectionResult(
                is_antonym=is_antonym,
                confidence=confidence,
                method="standard_model",
                evidence=f"Standard model distance: {semantic_distance:.3f}",
                semantic_distance=semantic_distance,
                context_similarity=0.0,
                model_scores={"standard_model": semantic_distance}
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="standard_model",
                evidence=f"Error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
    
    def _validate_known_patterns(self, text1: str, text2: str) -> AntonymDetectionResult:
        """Validate against known antonym patterns"""
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
                    context_similarity=0.0,
                    model_scores={"known_pattern": 1.0}
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
                context_similarity=0.0,
                model_scores={"negation_pattern": 0.8}
            )
        
        return AntonymDetectionResult(
            is_antonym=False,
            confidence=AntonymConfidence.NONE,
            method="known_pattern",
            evidence="No known patterns matched",
            semantic_distance=0.0,
            context_similarity=0.0,
            model_scores={}
        )
    
    def _detect_negation_pattern(self, text1: str, text2: str) -> bool:
        """Detect if texts differ by negation prefixes"""
        for prefix in self._negation_prefixes:
            if text1.startswith(prefix) and text1[len(prefix):] == text2:
                return True
            if text2.startswith(prefix) and text2[len(prefix):] == text1:
                return True
        return False
    
    def _needs_llm_validation(self, *results) -> bool:
        """Determine if LLM validation is needed based on conflicting results"""
        antonym_votes = sum(1 for r in results if r and r.is_antonym)
        total_votes = sum(1 for r in results if r and r.confidence != AntonymConfidence.NONE)
        
        # Use LLM if results are conflicting (close to 50/50 split)
        return total_votes > 0 and 0.3 <= antonym_votes / total_votes <= 0.7
    
    def _llm_validate_antonyms(self, text1: str, text2: str, context: str) -> Optional[AntonymDetectionResult]:
        """Use LLM to validate antonym relationship for complex cases"""
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
                context_similarity=0.0,
                model_scores={"llm_validation": 1.0 if is_antonym else 0.0}
            )
            
        except Exception as e:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="llm_validation",
                evidence=f"LLM error: {str(e)}",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
    
    def _combine_detection_results(self, *results: AntonymDetectionResult) -> AntonymDetectionResult:
        """Combine multiple detection results using advanced weighted scoring"""
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            return AntonymDetectionResult(
                is_antonym=False,
                confidence=AntonymConfidence.NONE,
                method="combined",
                evidence="No valid results",
                semantic_distance=0.0,
                context_similarity=0.0,
                model_scores={}
            )
        
        # Advanced weights based on model reliability
        weights = {
            "zero_shot": 0.4,        # Highest weight for specialized antonym detection
            "negation_model": 0.3,   # Good for negation patterns
            "known_pattern": 0.2,    # Reliable but limited coverage
            "standard_model": 0.1,   # Fallback method
            "llm_validation": 0.3,   # High weight when available
            "negation_pattern": 0.2  # Rule-based fallback
        }
        
        # Calculate weighted scores
        total_weight = 0.0
        weighted_antonym_score = 0.0
        weighted_confidence_score = 0.0
        
        evidence_parts = []
        semantic_distances = []
        context_similarities = []
        all_model_scores = {}
        
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
            
            # Merge model scores
            all_model_scores.update(result.model_scores)
        
        # Normalize scores
        if total_weight > 0:
            final_antonym_score = weighted_antonym_score / total_weight
            final_confidence_score = weighted_confidence_score / total_weight
        else:
            final_antonym_score = 0.0
            final_confidence_score = 0.0
        
        # Determine final result with improved logic
        is_antonym = final_antonym_score > 0.4  # Lower threshold for better recall
        
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
            context_similarity=avg_context_similarity,
            model_scores=all_model_scores
        )
    
    def batch_detect_antonyms(self, text_pairs: List[Tuple[str, str]], 
                            contexts: List[str] = None) -> List[AntonymDetectionResult]:
        """Detect antonyms for multiple text pairs efficiently"""
        if contexts is None:
            contexts = [""] * len(text_pairs)
        
        results = []
        for i, (text1, text2) in enumerate(text_pairs):
            context = contexts[i] if i < len(contexts) else ""
            result = self.detect_antonyms(text1, text2, context)
            results.append(result)
        
        return results
