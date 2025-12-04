"""
NLI (Natural Language Inference) service for semantic grading

HYBRID APPROACH with Sequential Refinement:
- Start with cosine similarity as initial score (x₁)
- Adjust score based on NLI-small results (x₂)
- Escalate to NLI-base if disagreement detected (x₃)
- Use LLM arbiter for critical cases (x₄)

This service uses transformer models to understand:
- Semantic entailment (does answer support the key point?)
- Contradictions (does answer negate the key point?)
- Paraphrasing and synonyms
- Negations and antonyms (built into the model)

Eliminates need for separate antonym/negation filters.
"""

from typing import Dict, List, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from core.config import settings


class NLIService:
    """
    Service for Natural Language Inference-based answer grading
    
    HYBRID 4-TRACK ARCHITECTURE:
    - Track 1 (Fast): Cosine ≥0.92 → instant pass, no NLI needed
    - Track 2 (Standard): Run NLI-small, check agreement with cosine
    - Track 3 (Deep): Run NLI-base if disagreement >15%
    - Track 4 (Critical): Trigger LLM arbiter if disagreement >55%
    
    Benefits:
    - Understands paraphrasing and synonyms
    - Detects contradictions and negations natively
    - No need for word-based filters
    - Sequential refinement for accurate scores
    - Multi-model voting reduces bias
    """
    
    def __init__(self):
        """Initialize NLI models (small for primary, base for escalation)"""
        # Primary model (fast, handles 95% of cases)
        self.model_small_name = getattr(settings.grading, 'nli_model_small', 'cross-encoder/nli-deberta-v3-small')
        # Escalation model (slower, for disagreements)
        self.model_base_name = getattr(settings.grading, 'nli_model_base', 'cross-encoder/nli-deberta-v3-base')
        
        # Initialize models
        self.model_small = None
        self.tokenizer_small = None
        self.model_base = None
        self.tokenizer_base = None
        
        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load primary model (small - always loaded)
        self._load_primary_model()
        
        # Base model loaded on-demand for efficiency
        self._base_model_loaded = False
        
        # Thresholds for hybrid approach
        self.fast_track_threshold = getattr(settings.grading, 'fast_track_cosine_high', 0.92)
        self.agreement_threshold = getattr(settings.grading, 'agreement_threshold', 0.15)
        self.moderate_disagreement = getattr(settings.grading, 'moderate_disagreement', 0.45)
        self.high_disagreement = getattr(settings.grading, 'high_disagreement', 0.55)
    
    def _load_primary_model(self):
        """Load the primary (small) NLI model"""
        print(f"📥 Loading primary NLI model: {self.model_small_name}")
        
        try:
            if not self.model_small_name or not isinstance(self.model_small_name, str):
                raise ValueError("Invalid model name: must be a non-empty string")
                
            self.tokenizer_small = AutoTokenizer.from_pretrained(self.model_small_name, use_fast=True)
            self.model_small = AutoModelForSequenceClassification.from_pretrained(self.model_small_name)
            self.model_small.eval()
            self.model_small.to(self.device)
            
            print(f"✅ Primary NLI model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load primary NLI model: {e}")
            print("⚠️ Falling back to embedding-only grading")
            self.model_small = None
    
    def _load_base_model(self):
        """Load the base model on-demand for escalation cases"""
        if self._base_model_loaded:
            return
        
        print(f"📥 Loading escalation NLI model: {self.model_base_name}")
        
        try:
            self.tokenizer_base = AutoTokenizer.from_pretrained(self.model_base_name, use_fast=True)
            self.model_base = AutoModelForSequenceClassification.from_pretrained(self.model_base_name)
            self.model_base.eval()
            self.model_base.to(self.device)
            self._base_model_loaded = True
            
            print(f"✅ Escalation NLI model loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️ Failed to load base model: {e}, using small model only")
            self.model_base = self.model_small
            self.tokenizer_base = self.tokenizer_small
            self._base_model_loaded = True
    
    @property
    def model(self):
        """Backward compatibility - return primary model"""
        return self.model_small
    
    def check_entailment(self, premise: str, hypothesis: str, use_base_model: bool = False) -> Dict[str, float]:
        """
        Check if hypothesis (student answer) is entailed by premise (key point)
        
        This is the core NLI function that replaces simple cosine similarity:
        - Understands paraphrasing: "big" vs "large" 
        - Detects negations: "not increasing" vs "increasing"
        - Catches contradictions: "hot" vs "cold"
        - Handles synonyms: "begin" vs "start"
        
        Args:
            premise: The reference text (key point from answer key)
            hypothesis: The text to check (student's answer or sentence)
            use_base_model: If True, use the larger base model for verification
            
        Returns:
            Dictionary with:
            - entailment: probability that hypothesis is entailed by premise (0-1)
            - neutral: probability of neutral relationship (0-1)
            - contradiction: probability of contradiction (0-1)
        
        Example:
            premise = "Inflation is the increase in prices"
            hypothesis = "Prices go up during inflation"
            Returns: {entailment: 0.92, neutral: 0.06, contradiction: 0.02}
            
            hypothesis = "Prices decrease during inflation"  
            Returns: {entailment: 0.01, neutral: 0.05, contradiction: 0.94}
        """
        # Select model based on parameter
        if use_base_model:
            self._load_base_model()  # Load on-demand
            model = self.model_base
            tokenizer = self.tokenizer_base
        else:
            model = self.model_small
            tokenizer = self.tokenizer_small
        
        if model is None:
            # Fallback if model failed to load
            return {"entailment": 0.5, "neutral": 0.5, "contradiction": 0.0}
        
        try:
            # Tokenize input pair
            inputs = tokenizer(
                premise, 
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(settings.grading, 'nli_max_length', 256),
                padding=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1)[0]
            
            # Cross-encoder NLI models output: [contradiction, neutral, entailment]
            return {
                "contradiction": float(probs[0]),
                "neutral": float(probs[1]),
                "entailment": float(probs[2])
            }
                
        except Exception as e:
            print(f"⚠️ NLI inference error: {e}")
            return {"entailment": 0.5, "neutral": 0.5, "contradiction": 0.0}
    
    def adjust_score(self, current_score: float, nli_result: Dict[str, float], has_contradiction: bool = False, has_explicit_negation: bool = False) -> Tuple[float, float]:
        """
        SEQUENTIAL REFINEMENT: Adjust the current score based on NLI evidence
        
        This implements the hybrid approach where we start with cosine as x₁
        and progressively refine based on NLI evidence.
        
        CRITICAL: Explicit negations (no, not, never, worse) ALWAYS trigger penalty.
        NLI-only contradictions are less reliable and get smaller penalty.
        
        Args:
            current_score: Current score (cosine or previously adjusted)
            nli_result: NLI result with entailment/contradiction/neutral
            has_contradiction: Whether contradiction was detected
            has_explicit_negation: Whether explicit negation words were found
            
        Returns:
            Tuple of (new_score, confidence)
            - new_score: Adjusted score (0-1)
            - confidence: How confident NLI is in this adjustment (0-1)
        """
        entailment = nli_result['entailment']
        contradiction = nli_result['contradiction']
        neutral = nli_result.get('neutral', 1 - entailment - contradiction)
        
        # Calculate NLI confidence (how decisive is the NLI result?)
        confidence = abs(entailment - contradiction)  # High when NLI is certain
        
        # CASE 1: EXPLICIT NEGATION - Always penalize heavily
        # This catches "no jobs", "not good", "never", "worse than", etc.
        if has_explicit_negation:
            penalty = 0.50  # Heavy penalty for explicit negation
            new_score = max(0.0, current_score - penalty)
            print(f"         ⛔ Explicit negation penalty: {current_score:.3f} → {new_score:.3f}")
            return new_score, 0.95  # High confidence in this penalty
        
        # CASE 2: NLI-only contradiction (no explicit negation) - Be more conservative
        # Only penalize if NLI is very confident (>0.75 contradiction)
        if has_contradiction and contradiction > 0.75 and entailment < 0.20:
            penalty = 0.30  # Moderate penalty for NLI-only contradiction
            new_score = max(0.0, current_score - penalty)
            print(f"         ⚠️ NLI contradiction penalty: {current_score:.3f} → {new_score:.3f} (con={contradiction:.3f})")
            return new_score, confidence
        
        # CASE 3: High entailment AND low contradiction → NLI confirms
        if entailment > 0.75 and contradiction < 0.30:
            # NLI strongly confirms - use weighted average favoring NLI
            new_score = current_score * 0.3 + entailment * 0.7
            
        # CASE 4: Moderate entailment → Slight adjustment toward NLI
        elif entailment > 0.50 and contradiction < 0.40:
            # Blend scores: 60% current, 40% NLI entailment
            new_score = current_score * 0.6 + entailment * 0.4
            
        # CASE 5: Low entailment or uncertain → Be conservative
        else:
            if entailment < 0.40 and contradiction < 0.40:
                # Neutral zone - small penalty for uncertainty
                new_score = current_score * 0.90
            else:
                # Mixed signals - trust cosine more
                new_score = current_score * 0.95
        
        # Clamp to valid range
        new_score = max(0.0, min(1.0, new_score))
        
        return new_score, confidence
    
    def hybrid_evaluate(
        self,
        student_answer: str,
        key_point: str,
        cosine_score: float,
        sentences: List[str] = None
    ) -> Dict[str, any]:
        """
        MAIN HYBRID EVALUATION: 4-Track routing with sequential refinement
        
        This is the core hybrid method that replaces the old evaluate_answer_against_keypoint.
        
        Args:
            student_answer: Full student answer text
            key_point: The key point to check for
            cosine_score: Initial cosine similarity score (x₁)
            sentences: Pre-split sentences (optional)
            
        Returns:
            Dictionary with:
            - track: Which track was used (1-4)
            - final_score: The refined score
            - is_covered: boolean if key point is covered
            - confidence: Overall confidence
            - nli_scores: NLI model outputs
            - disagreement: Cosine vs NLI disagreement
            - needs_llm: Whether LLM arbiter should be triggered
        """
        # Track 1: FAST - High cosine similarity, skip NLI
        if cosine_score >= self.fast_track_threshold:
            return {
                "track": 1,
                "track_name": "fast",
                "final_score": cosine_score,
                "is_covered": True,
                "confidence": 0.95,
                "nli_scores": None,
                "disagreement": 0.0,
                "needs_llm": False,
                "details": f"Fast track: cosine {cosine_score:.3f} ≥ {self.fast_track_threshold}"
            }
        
        # Track 1b: Very low cosine, likely wrong - skip NLI
        if cosine_score < 0.55:
            return {
                "track": 1,
                "track_name": "fast-fail",
                "final_score": cosine_score,
                "is_covered": False,
                "confidence": 0.90,
                "nli_scores": None,
                "disagreement": 0.0,
                "needs_llm": False,
                "details": f"Fast fail: cosine {cosine_score:.3f} < 0.55"
            }
        
        # Prepare sentences for NLI
        if sentences is None:
            sentences = [s.strip() for s in student_answer.split('.') if s.strip()]
            if not sentences:
                sentences = [student_answer]
        
        # Track 2+: Run NLI-small
        nli_small_result = self._evaluate_sentences_nli(sentences, key_point, use_base=False)
        
        # Calculate adjusted score (sequential refinement)
        # CRITICAL: Pass both has_contradiction AND has_explicit_negation flags
        x2, conf_small = self.adjust_score(cosine_score, {
            "entailment": nli_small_result["best_entailment"],
            "contradiction": nli_small_result["max_contradiction"],
            "neutral": 1 - nli_small_result["best_entailment"] - nli_small_result["max_contradiction"]
        }, has_contradiction=nli_small_result["has_contradiction"],
           has_explicit_negation=nli_small_result.get("has_explicit_negation", False))
        
        disagreement = abs(x2 - cosine_score)
        
        # If contradiction detected, force Track 3/4 for deeper verification
        # This prevents false positives where cosine is high but NLI finds contradiction
        if nli_small_result["has_contradiction"]:
            print(f"         🔄 Contradiction found by NLI-small, forcing deep verification")
        
        # Track 2: STANDARD - Models agree (disagreement < 15%) AND no contradiction
        if disagreement < self.agreement_threshold and conf_small > 0.60 and not nli_small_result["has_contradiction"]:
            # Use threshold to determine if covered
            threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.50)
            
            # ADDITIONAL CHECK: Require minimum specificity
            # If cosine score is low (<0.80), require higher NLI entailment (>0.70)
            # This prevents vague phrases like "causes problems" from matching specific concepts
            if cosine_score < 0.80 and nli_small_result["best_entailment"] < 0.70:
                # Vague match - require more evidence
                is_covered = False
                print(f"         ⚠️ Vague match rejected: cosine={cosine_score:.3f} < 0.80, entailment={nli_small_result['best_entailment']:.3f} < 0.70")
            else:
                is_covered = x2 >= 0.65 and nli_small_result["best_entailment"] >= threshold and not nli_small_result["has_contradiction"]
            
            return {
                "track": 2,
                "track_name": "standard",
                "final_score": x2,
                "is_covered": is_covered,
                "confidence": conf_small,
                "nli_scores": {
                    "small": nli_small_result
                },
                "disagreement": disagreement,
                "needs_llm": False,
                "details": f"Standard: agree ({disagreement:.3f} < {self.agreement_threshold}), refined {cosine_score:.3f}→{x2:.3f}"
            }
        
        # Track 3: DEEP VERIFY - Moderate disagreement (15-55%)
        # Load and run base model for verification
        nli_base_result = self._evaluate_sentences_nli(sentences, key_point, use_base=True)
        
        # Refine again with base model
        # CRITICAL: Pass both has_contradiction AND has_explicit_negation flags
        x3, conf_base = self.adjust_score(x2, {
            "entailment": nli_base_result["best_entailment"],
            "contradiction": nli_base_result["max_contradiction"],
            "neutral": 1 - nli_base_result["best_entailment"] - nli_base_result["max_contradiction"]
        }, has_contradiction=nli_base_result["has_contradiction"],
           has_explicit_negation=nli_base_result.get("has_explicit_negation", False))
        
        # Average confidence from both models
        combined_confidence = (conf_small + conf_base) / 2
        
        # Check if models vote similarly (agreement between small and base)
        model_agreement = abs(nli_small_result["best_entailment"] - nli_base_result["best_entailment"]) < 0.20
        
        # Track 3: Moderate disagreement but models agree on NLI
        if disagreement < self.high_disagreement or model_agreement:
            threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.50)
            # Use average of both NLI entailments for decision
            avg_entailment = (nli_small_result["best_entailment"] + nli_base_result["best_entailment"]) / 2
            has_any_contradiction = nli_small_result["has_contradiction"] or nli_base_result["has_contradiction"]
            
            is_covered = x3 >= 0.60 and avg_entailment >= threshold and not has_any_contradiction
            
            return {
                "track": 3,
                "track_name": "deep",
                "final_score": x3,
                "is_covered": is_covered,
                "confidence": combined_confidence,
                "nli_scores": {
                    "small": nli_small_result,
                    "base": nli_base_result
                },
                "disagreement": disagreement,
                "needs_llm": False,
                "details": f"Deep verify: refined {cosine_score:.3f}→{x2:.3f}→{x3:.3f}, model_agree={model_agreement}"
            }
        
        # Track 4: CRITICAL - High disagreement, need LLM arbiter
        threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.50)
        avg_entailment = (nli_small_result["best_entailment"] + nli_base_result["best_entailment"]) / 2
        has_any_contradiction = nli_small_result["has_contradiction"] or nli_base_result["has_contradiction"]
        
        # Tentative decision (may be overridden by LLM)
        tentative_covered = x3 >= 0.55 and avg_entailment >= threshold and not has_any_contradiction
        
        return {
            "track": 4,
            "track_name": "critical",
            "final_score": x3,
            "is_covered": tentative_covered,
            "confidence": combined_confidence,
            "nli_scores": {
                "small": nli_small_result,
                "base": nli_base_result
            },
            "disagreement": disagreement,
            "needs_llm": True,  # Flag for LLM arbiter
            "details": f"Critical: high disagreement ({disagreement:.3f} ≥ {self.high_disagreement}), LLM recommended"
        }
    
    def _evaluate_sentences_nli(self, sentences: List[str], key_point: str, use_base: bool = False) -> Dict[str, any]:
        """
        Evaluate all sentences against a key point using NLI
        
        Args:
            sentences: List of sentences from student answer
            key_point: The key point to check
            use_base: Whether to use base model
            
        Returns:
            Aggregated NLI results
        """
        best_entailment = 0.0
        max_contradiction = 0.0
        has_contradiction = False
        has_explicit_negation = False
        details = []
        
        # EXPLICIT NEGATION PATTERNS - these override NLI when detected
        # Split into two categories:
        # 1. Pure negators (always negative): "no", "not", "never", etc.
        # 2. Context-dependent: REMOVED - these caused too many false positives
        #    (e.g., "overcrowding" in sentence about effects was wrongly flagged 
        #     when checking education key point just because "urban" appeared)
        
        pure_negators = [
            "no ", "not ", "don't ", "doesn't ", "didn't ", "won't ", "wouldn't ",
            "never ", "none ", "nothing ", "nobody ", "nowhere ",
            "isn't ", "aren't ", "wasn't ", "weren't ", "hasn't ", "haven't ",
            "can't ", "cannot ", "couldn't ", "shouldn't ",
            "opposite", "contrary", "reverse"
        ]
        
        key_point_lower = key_point.lower()
        # Extract CORE concepts (longer, more meaningful words)
        key_concepts = [w for w in key_point_lower.split() if len(w) > 4]
        
        for sentence in sentences:
            if len(sentence.strip()) < 5:
                continue
            
            sentence_lower = sentence.lower()
            
            # Check for PURE negation patterns (always bad)
            # These are words like "no", "not", "never" that clearly negate
            sentence_has_negation = False
            for pattern in pure_negators:
                if pattern in sentence_lower:
                    # Check if negation is near a CORE key concept (not just any word)
                    for concept in key_concepts:
                        if concept in sentence_lower:
                            # Pure negation + key concept = likely contradiction
                            sentence_has_negation = True
                            has_explicit_negation = True
                            print(f"         🔴 Explicit negation detected: '{pattern.strip()}' near '{concept}'")
                            break
                if sentence_has_negation:
                    break
            
            scores = self.check_entailment(
                premise=sentence,
                hypothesis=key_point,
                use_base_model=use_base
            )
            
            details.append({
                "sentence": sentence,
                "has_explicit_negation": sentence_has_negation,
                **scores
            })
            
            if scores["entailment"] > best_entailment and not sentence_has_negation:
                best_entailment = scores["entailment"]
            
            if scores["contradiction"] > max_contradiction:
                max_contradiction = scores["contradiction"]
            
            # IMPROVED contradiction detection:
            # 1. Explicit negation pattern detected, OR
            # 2. NLI high contradiction with low entailment
            if sentence_has_negation:
                has_contradiction = True
                print(f"         🚨 Contradiction (explicit negation): '{sentence[:50]}...'")
            elif (scores["contradiction"] > 0.60 and 
                  scores["entailment"] < 0.35 and 
                  (scores["contradiction"] - scores["entailment"]) > 0.30):
                has_contradiction = True
                print(f"         🚨 Contradiction (NLI): ent={scores['entailment']:.3f}, con={scores['contradiction']:.3f}")
        
        return {
            "best_entailment": best_entailment,
            "max_contradiction": max_contradiction,
            "has_contradiction": has_contradiction,
            "has_explicit_negation": has_explicit_negation,
            "details": details
        }
    
    def evaluate_answer_against_keypoint(
        self, 
        student_answer: str, 
        key_point: str,
        sentences: List[str] = None,
        cosine_score: float = None
    ) -> Dict[str, any]:
        """
        BACKWARD COMPATIBLE: Evaluate if student answer covers a key point
        
        Now uses hybrid_evaluate internally when cosine_score is provided.
        Falls back to old behavior for compatibility.
        
        Args:
            student_answer: Full student answer text
            key_point: The key point to check for
            sentences: Pre-split sentences (optional)
            cosine_score: If provided, uses hybrid approach
            
        Returns:
            Dictionary with:
            - is_covered: boolean if key point is covered
            - confidence: confidence score (0-1)
            - best_entailment: highest entailment score
            - has_contradiction: boolean if any contradiction found
            - details: per-sentence scores
            - track: (hybrid only) which track was used
        """
        # If cosine_score provided, use new hybrid approach
        if cosine_score is not None:
            hybrid_result = self.hybrid_evaluate(
                student_answer=student_answer,
                key_point=key_point,
                cosine_score=cosine_score,
                sentences=sentences
            )
            
            # Map to old return format for compatibility
            nli_scores = hybrid_result.get("nli_scores", {})
            small_scores = nli_scores.get("small", {}) if nli_scores else {}
            
            return {
                "is_covered": hybrid_result["is_covered"],
                "confidence": hybrid_result["confidence"],
                "best_entailment": small_scores.get("best_entailment", hybrid_result["final_score"]),
                "has_contradiction": small_scores.get("has_contradiction", False),
                "details": small_scores.get("details", []),
                "track": hybrid_result["track"],
                "track_name": hybrid_result["track_name"],
                "final_score": hybrid_result["final_score"],
                "disagreement": hybrid_result["disagreement"],
                "needs_llm": hybrid_result["needs_llm"]
            }
        
        # Old behavior (no cosine score - pure NLI mode)
        if self.model_small is None:
            return {
                "is_covered": False,
                "confidence": 0.0,
                "best_entailment": 0.0,
                "has_contradiction": False,
                "details": []
            }
        
        # Split into sentences if not provided
        if sentences is None:
            sentences = [s.strip() for s in student_answer.split('.') if s.strip()]
            if not sentences:
                sentences = [student_answer]
        
        # Use the internal method
        result = self._evaluate_sentences_nli(sentences, key_point, use_base=False)
        
        # Determine if covered
        entailment_threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.50)
        is_covered = (
            result["best_entailment"] >= entailment_threshold and 
            not result["has_contradiction"]
        )
        
        confidence = 0.0 if result["has_contradiction"] else result["best_entailment"]
        
        return {
            "is_covered": is_covered,
            "confidence": confidence,
            "best_entailment": result["best_entailment"],
            "has_contradiction": result["has_contradiction"],
            "details": result["details"]
        }
    
    def batch_evaluate_keypoints(
        self,
        student_answer: str,
        key_points: List[str],
        sentences: List[str] = None,
        cosine_scores: List[float] = None
    ) -> List[Dict[str, any]]:
        """
        Evaluate multiple key points against student answer
        
        Args:
            student_answer: Full student answer
            key_points: List of key point texts to check
            sentences: Pre-split sentences (optional, for efficiency)
            cosine_scores: List of cosine scores for hybrid mode (optional)
            
        Returns:
            List of evaluation results, one per key point
        """
        results = []
        
        for i, key_point in enumerate(key_points):
            cosine = cosine_scores[i] if cosine_scores and i < len(cosine_scores) else None
            
            result = self.evaluate_answer_against_keypoint(
                student_answer=student_answer,
                key_point=key_point,
                sentences=sentences,
                cosine_score=cosine
            )
            result["key_point"] = key_point
            results.append(result)
        
        return results
    
    def explain_score(self, nli_result: Dict[str, any]) -> str:
        """
        Generate human-readable explanation of NLI score
        
        Args:
            nli_result: Result from check_entailment or hybrid_evaluate
            
        Returns:
            Explanation string
        """
        # Handle hybrid result format
        if "track_name" in nli_result:
            track = nli_result["track_name"]
            score = nli_result.get("final_score", 0)
            
            if track == "fast":
                return f"✅ High similarity ({score:.2f}) - instant pass"
            elif track == "fast-fail":
                return f"❌ Low similarity ({score:.2f}) - answer doesn't match"
            elif track == "standard":
                return f"✓ Verified by NLI ({score:.2f})"
            elif track == "deep":
                return f"🔍 Deep verification ({score:.2f})"
            elif track == "critical":
                return f"⚠️ Requires careful review ({score:.2f})"
        
        # Old format
        entailment = nli_result.get("entailment", nli_result.get("best_entailment", 0))
        contradiction = nli_result.get("contradiction", nli_result.get("has_contradiction", False))
        
        if isinstance(contradiction, bool):
            if contradiction:
                return "❌ Your answer contradicts the key point"
        elif contradiction > 0.7:
            return "❌ Your answer contradicts the key point"
        
        if entailment > 0.8:
            return "✅ Your answer fully covers this point"
        elif entailment > 0.6:
            return "✓ Your answer partially covers this point"
        elif entailment > 0.4:
            return "~ Your answer somewhat relates to this point"
        else:
            return "❌ Your answer does not cover this point"
