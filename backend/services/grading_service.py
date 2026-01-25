"""
Answer grading service using embeddings and text analysis
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from fastapi import HTTPException
import json
from datetime import datetime
from pathlib import Path

from models.models import AnswerResponse
from services.embedding_service import EmbeddingService
from services.text_processing import TextProcessor
from services.question_service import QuestionService
from services.embedding_storage import EmbeddingStorage
from services.answer_cache_service import AnswerCacheService
from services.smart_antonym_detector import SmartAntonymDetector, AntonymConfidence
from services.verification import LLMVerificationService, LLMVerificationResult
from core.config import settings

# Conditional NLI import (only for hybrid/nli modes)
if getattr(settings.grading, 'grading_method', 'embedding') in ["hybrid", "nli"]:
    from services.nli_service import NLIService

# Conditional LLM arbiter import (only if enabled)
if getattr(settings.grading, 'llm_arbiter_enabled', False):
    from services.llm_arbiter_service import LLMArbiterService


class GradingService:
    """
    Service for grading user answers against question key points
    
    This service handles:
    - Precomputing embeddings for key points
    - Grading user answers using semantic similarity
    - Combining semantic and lexical analysis
    - Generating feedback based on grading results
    - Logging uncertain cases for active learning
    """
    
    def __init__(self, question_service: QuestionService, openai_client):
        """
        Initialize grading service with dependencies
        
        Args:
            question_service: Service for managing questions
            openai_client: OpenAI client instance for API calls
        """
        self._question_service = question_service
        self._embedding_service = EmbeddingService(openai_client)
        self._text_processor = TextProcessor()
        self._embedding_storage = EmbeddingStorage()
        self._answer_cache_service = AnswerCacheService(
            self._embedding_service,
            self._embedding_storage._index
        )
        self._ai_antonym_detector = SmartAntonymDetector(self._embedding_service, openai_client)
        self._llm_verification = (
            LLMVerificationService(openai_client) if openai_client else None
        )
        
        # Initialize NLI service for hybrid/nli modes
        self._nli_service = None
        if getattr(settings.grading, 'grading_method', 'embedding') in ["hybrid", "nli"]:
            print("Initializing NLI service for hybrid grading...")
            self._nli_service = NLIService()
        
        # Initialize LLM arbiter for high-uncertainty cases (optional, ultra-low-cost)
        self._llm_arbiter = None
        if getattr(settings.grading, 'llm_arbiter_enabled', False):
            self._llm_arbiter = LLMArbiterService()
        
        # Storage for precomputed embeddings and keywords
        self._key_point_embeddings: Dict[int, List[List[float]]] = {}
        self._key_point_keywords: Dict[int, List[Set[str]]] = {}
        
        # Configuration
        self._similarity_config = settings.grading.similarity_thresholds
        self._feedback_config = settings.grading.feedback_messages
        self._validation_config = settings.grading.answer_validation
        self._antonym_config = settings.antonym_detection
        
        # Uncertainty logging for active learning (zero cost)
        self._log_uncertain = getattr(settings.grading, 'log_uncertain_cases', False)
        self._uncertainty_threshold = getattr(settings.grading, 'uncertainty_threshold', 0.7)
        self._uncertain_cases_file = Path("uncertain_cases.jsonl")
        
        # Hybrid mode thresholds - NEW 4-TRACK APPROACH
        if getattr(settings.grading, 'grading_method', 'embedding') == "hybrid":
            # Fast track (Track 1)
            self._fast_track_threshold = getattr(settings.grading, 'fast_track_cosine_high', 0.92)
            # Standard track bounds (Track 2)
            self._standard_track_min = getattr(settings.grading, 'standard_track_min', 0.70)
            # Disagreement thresholds for escalation
            self._agreement_threshold = getattr(settings.grading, 'agreement_threshold', 0.15)
            self._high_disagreement = getattr(settings.grading, 'high_disagreement', 0.55)
            
            print(f"🚀 HYBRID 4-TRACK GRADING enabled:")
            print(f"   Track 1 (Fast): cosine ≥ {self._fast_track_threshold} → instant pass")
            print(f"   Track 2 (Standard): {self._standard_track_min}-{self._fast_track_threshold} → NLI-small verify")
            print(f"   Track 3 (Deep): disagreement > {self._agreement_threshold} → NLI-base verify")
            print(f"   Track 4 (Critical): disagreement > {self._high_disagreement} → LLM arbiter")
            if self._log_uncertain:
                print(f"   Uncertainty logging: enabled (threshold={self._uncertainty_threshold})")
    
    def _calculate_uncertainty_score(
        self, 
        similarity: float, 
        nli_entailment: float = None,
        nli_contradiction: float = None,
        overlap: float = 0.0
    ) -> float:
        """
        Calculate uncertainty score for active learning (0-1, higher = more uncertain)
        
        Cases with high uncertainty are logged for human review to improve the system.
        This is a zero-cost approach to continuous improvement.
        """
        # Factor 1: Decision boundary proximity
        boundary_70 = abs(similarity - 0.70)
        boundary_85 = abs(similarity - 0.85)
        boundary_uncertainty = max(0, 1 - min(boundary_70, boundary_85) / 0.15)
        
        # Factor 2: Cosine-NLI disagreement
        disagreement = 0
        if nli_entailment is not None:
            disagreement = abs(similarity - nli_entailment)
        
        # Factor 3: NLI confidence (if available)
        nli_uncertainty = 0
        if nli_entailment is not None and nli_contradiction is not None:
            nli_max = max(nli_entailment, nli_contradiction)
            nli_uncertainty = 1 - nli_max
        
        # Factor 4: Cosine-overlap mismatch
        cosine_overlap_mismatch = 0
        if similarity > 0.75 and overlap < 0.3:
            cosine_overlap_mismatch = similarity - overlap - 0.3
        
        # Weighted combination
        uncertainty = (
            0.40 * boundary_uncertainty +
            0.30 * disagreement +
            0.20 * nli_uncertainty +
            0.10 * cosine_overlap_mismatch
        )
        
        return min(1.0, uncertainty)
    
    def _log_uncertain_case(
        self,
        question_id: int,
        question_text: str,
        key_point: str,
        student_answer: str,
        similarity: float,
        nli_result: Dict = None,
        overlap: float = 0.0,
        decision: bool = False,
        uncertainty_score: float = 0.0
    ) -> None:
        """
        Log uncertain case for human review (active learning - zero cost)
        
        Logged cases can be periodically reviewed to:
        - Validate system decisions
        - Find patterns in errors
        - Fine-tune thresholds
        - Improve NLI model
        """
        if not self._log_uncertain:
            return
        
        case = {
            "timestamp": datetime.now().isoformat(),
            "question_id": question_id,
            "question_text": question_text,
            "key_point": key_point,
            "student_answer": student_answer,
            "similarity": round(similarity, 3),
            "overlap": round(overlap, 3),
            "nli_entailment": round(nli_result.get("best_entailment", 0), 3) if nli_result else None,
            "nli_contradiction": nli_result.get("has_contradiction", False) if nli_result else None,
            "uncertainty_score": round(uncertainty_score, 3),
            "decision": decision,
            "needs_review": True
        }
        
        # Append to JSONL file (one JSON object per line)
        try:
            with open(self._uncertain_cases_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ Failed to log uncertain case: {e}")
    
    def precompute_embeddings(self) -> None:
        """
        Precompute or load embeddings for all key points
        
        Based on configuration:
        - If precompute_embeddings=True: compute fresh embeddings and save to cache
        - If precompute_embeddings=False: load from cache, compute if cache missing/invalid
        """
        all_questions = self._question_service.get_all_questions()
        
        # Create metadata for questions to validate cache consistency
        questions_metadata = self._create_questions_metadata(all_questions)
        
        # Check configuration and try to load from cache first
        if not settings.grading.precompute_embeddings:
            print("🔄 Attempting to load embeddings from cache...")
            
            loaded_data = self._embedding_storage.load_cached_embeddings(questions_metadata)
            if loaded_data is not None:
                self._key_point_embeddings, self._key_point_keywords = loaded_data
                print(f"✅ Loaded embeddings from cache for {len(all_questions)} questions")
                return
            else:
                print("⚠️ Cache load failed, no embeddings available")
                print("⚠️ Service will start but embeddings need to be computed later")
                print("💡 Add OpenAI credits and set precompute_embeddings=true to generate embeddings")
                return
        else:
            print("🔄 Precomputing fresh embeddings (precompute_embeddings=True)...")
        
        # Compute embeddings fresh
        try:
            self._compute_new_embeddings(all_questions)
            
            # Save to cache for future use
            print("💾 Saving embeddings to cache...")
        except Exception as e:
            print(f"❌ Failed to compute embeddings: {e}")
            print("⚠️ Service will start but embeddings need to be computed later")
            print("💡 Check your OpenAI quota and billing details")
            return
        success = self._embedding_storage.cache_embeddings(
            self._key_point_embeddings,
            self._key_point_keywords,
            questions_metadata
        )
        
        if not success:
            print("⚠️ Failed to save embeddings cache, but continuing with computed embeddings")
        
        print(f"✅ Embeddings ready for {len(all_questions)} questions")
    
    def _create_questions_metadata(self, questions: List[Dict]) -> Dict[int, Dict[str, Any]]:
        """
        Create metadata for questions to validate cache consistency
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            Dictionary mapping question_id to metadata
        """
        metadata = {}
        
        for question in questions:
            question_id = question["question_id"]
            metadata[question_id] = {
                "question_text": question["question_text"],
                "key_points_count": len(question["key_points"]),
                "key_points_texts": [kp["text"] for kp in question["key_points"]]
            }
        
        return metadata

    
    def _compute_new_embeddings(self, questions: List[Dict]) -> None:
        """
        Compute new embeddings for all key points
        
        Args:
            questions: List of question dictionaries
        """
        print("🔄 Computing fresh embeddings for key points...")
        
        for question in questions:
            question_id = question["question_id"]
            embeddings = []
            keywords_list: List[Set[str]] = []
            
            for key_point in question["key_points"]:
                text = key_point["text"]
                
                # Get embedding for key point
                embedding = self._embedding_service.get_embedding(text)
                embeddings.append(embedding)
                
                # Process keywords for lexical matching
                keywords = set(self._text_processor.normalize_text(text))
                keywords_list.append(keywords)
                
                print(f"  ✅ Embedded: '{text[:30]}...'")
            
            self._key_point_embeddings[question_id] = embeddings
            self._key_point_keywords[question_id] = keywords_list
            
            print(f"  📝 Question {question_id}: {len(embeddings)} key points embedded")
    
    def validate_answer(self, user_answer: str) -> Optional[AnswerResponse]:
        """
        Validate user answer for basic requirements
        
        Args:
            user_answer: The user's answer text
            
        Returns:
            AnswerResponse with error feedback if invalid, None if valid
        """
        user_answer_clean = user_answer.strip()
        
        # Check for very short answers
        if (len(user_answer_clean) < self._validation_config.min_answer_length or 
            len(user_answer_clean.split()) < self._validation_config.min_word_count):
            return AnswerResponse(
                score=0.0,
                hit_key_points=[],
                missing_key_points=[],
                feedback=self._feedback_config.short_answer
            )
        
        # Check for violent answers
        if any(violent_answer in user_answer_clean.lower() for violent_answer in self._validation_config.violent_answers):
            return AnswerResponse(
                score=0.0,
                hit_key_points=[],
                missing_key_points=[],
                feedback=settings.grading.answer_validation.feedback_message.format(invalid_answer=user_answer_clean)
            )

        # The answer is invalid if
        # it is similar to "i don't know" and similar answers
        user_answer_embedding = self._embedding_service.get_embedding(user_answer_clean)

        invalid_answers_similarities = []
        for invalid_answer in self._validation_config.invalid_answers:
            invalid_answer_embedding = self._embedding_service.get_embedding(invalid_answer)
            similarity = self._embedding_service.compute_cosine_similarity(
                user_answer_embedding, 
                invalid_answer_embedding
            )
            invalid_answers_similarities.append(similarity)
        
        if any(similarity > 0.85 for similarity in invalid_answers_similarities):
            return AnswerResponse(
                score=0.0,
                hit_key_points=[],
                missing_key_points=[],
                feedback=settings.grading.answer_validation.feedback_message.format(invalid_answer=user_answer_clean)
            )
        
        return None  # Answer is valid
    
    def grade_answer(self, question_id: int, user_answer: str) -> AnswerResponse:
        """
        Grade user answer against key points using embedding similarity
        
        This is the core grading logic:
        1. Check answer cache for similar previously graded answers
        2. If cache hit, return cached grade immediately
        3. Otherwise, validate the answer
        4. Get user answer embeddings (sentence-level)
        5. Compare with each key point embedding using cosine similarity
        6. Mark key points as "hit" or "missing" based on similarity threshold
        7. Calculate score and generate feedback
        8. Cache the answer for future use
        
        Args:
            question_id: ID of the question being answered
            user_answer: The student's answer text
            
        Returns:
            AnswerResponse with score, hit/missing points, and feedback
        """
        # ==========================================
        # FAST PATH: CHECK ANSWER CACHE
        # ==========================================
        cached_result = self._answer_cache_service.retrieve_similar_cached_answers(
            question_id, user_answer
        )
        
        if cached_result is not None:
            _, cached_record = cached_result
            cached_grade = AnswerResponse(
                score=cached_record.get('score', 0),
                hit_key_points=cached_record.get('hit_key_points', []),
                missing_key_points=cached_record.get('missing_key_points', []),
                feedback=cached_record.get('feedback', '')
            )
            return cached_grade
        
        # ==========================================
        # NORMAL PATH: COMPUTE GRADE
        # ==========================================
        # Validate answer first
        validation_result = self.validate_answer(user_answer)
        if validation_result:
            return validation_result
        
        # Get question data
        question = self._question_service.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        key_points = question["key_points"]

        # If embeddings for this question are missing (cache was empty), compute them on-demand
        if question_id not in self._key_point_embeddings:
            print(f"⚠️ Embeddings missing for question {question_id}, computing on demand...")
            try:
                self._compute_embeddings_for_question(question)
            except Exception as e:
                print(f"❌ Failed to compute embeddings on demand for question {question_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to compute embeddings for evaluation")
        
        # Split user answer into sentences and get embeddings
        sentences = self._text_processor.split_into_sentences(user_answer)
        
        try:
            sentence_embeddings = self._embedding_service.get_batch_embeddings(sentences)
        except Exception:
            # Fallback to single embedding of full answer
            sentence_embeddings = [self._embedding_service.get_embedding(user_answer)]
            sentences = [user_answer]
        
        # Process user answer tokens for lexical matching
        user_tokens = self._text_processor.normalize_text(user_answer)
        
        # DETECT COMPREHENSIVE BALANCED ANSWERS
        # Answers discussing both positive and negative perspectives should NOT trigger contradiction checks
        # STRICTER: Require multiple indicators or specific phrases, not just single words
        balanced_phrases = [
            "on one hand", "on the other hand", "both positive and negative",
            "benefits and challenges", "advantages and disadvantages",
            "pros and cons", "both sides", "dual effect"
        ]
        # Require at least one phrase OR combination of "both" with perspective words
        user_lower = user_answer.lower()
        has_balanced_phrase = any(phrase in user_lower for phrase in balanced_phrases)
        has_both_perspectives = "both" in user_lower and any(w in user_lower for w in ["positive", "negative", "good", "bad"])
        is_comprehensive_answer = has_balanced_phrase or has_both_perspectives
        
        if is_comprehensive_answer:
            print(f"\n📝 COMPREHENSIVE ANSWER DETECTED: Discussing multiple perspectives")
            print(f"   → Skipping contradiction checks for individual sentences")
        
        # GLOBAL CONTRADICTION CHECK: Check if answer contradicts the question itself
        # This prevents false positives when student negates the main premise
        global_contradiction_detected = False
        if getattr(settings.grading, 'grading_method', 'embedding') == "hybrid" and self._nli_service:
            # Check the main question text against the answer for fundamental contradictions
            question_text = question["question_text"]
            global_nli_result = self._nli_service.evaluate_answer_against_keypoint(
                student_answer=user_answer,
                key_point=question_text,
                sentences=sentences
            )
            # Only flag global contradiction if NOT a comprehensive answer
            # (comprehensive answers naturally discuss negatives, which shouldn't be flagged)
            if not is_comprehensive_answer and global_nli_result["has_contradiction"] and global_nli_result["best_entailment"] < 0.3:
                global_contradiction_detected = True
                print(f"\n⚠️ GLOBAL CONTRADICTION: Answer contradicts the question premise")
                print(f"   Question: '{question_text}'")
                print(f"   Contradiction score: {global_nli_result['has_contradiction']}")
        
        # Evaluate each key point
        hit_key_points = []
        missing_key_points = []
        confidence_scores = []  # Track confidence for smart LLM fallback
        
        reference_chunks = [kp["text"] for kp in key_points]

        for i, key_point in enumerate(key_points):
            key_point_embedding = self._key_point_embeddings[question_id][i]
            key_point_tokens = list(self._key_point_keywords[question_id][i])
            key_point_text = key_point["text"]
            
            # Calculate semantic similarity (best sentence match)
            similarity, best_sentence = self._find_best_sentence_match(
                sentences, sentence_embeddings, key_point_embedding
            )
            raw_similarity = similarity
            
            # Calculate lexical overlap
            overlap = self._text_processor.calculate_token_overlap(
                user_tokens, key_point_tokens
            )
            raw_overlap = overlap
            
            # THREE-TIER HYBRID GRADING SYSTEM
            is_hit = False
            verification_method = ""
            hybrid_result = None  # Track for LLM fallback confidence
            
            # If global contradiction detected, auto-fail all key points
            if global_contradiction_detected:
                is_hit = False
                verification_method = "global-contradiction-fail"
                print(f"  ❌ '{key_point_text[:30]}...': FAILED (global contradiction)")
            
            elif getattr(settings.grading, 'grading_method', 'embedding') == "hybrid" and self._nli_service:
                # ===========================================
                # HYBRID 4-TRACK ROUTING WITH SEQUENTIAL REFINEMENT
                # ===========================================
                # Track 1 (Fast): cosine ≥0.92 → instant pass
                # Track 2 (Standard): NLI-small, check agreement
                # Track 3 (Deep): NLI-base if disagreement >15%
                # Track 4 (Critical): LLM arbiter if disagreement >55%
                # ===========================================
                
                # Use the new hybrid_evaluate method
                hybrid_result = self._nli_service.hybrid_evaluate(
                    student_answer=user_answer,
                    key_point=key_point_text,
                    cosine_score=similarity,
                    sentences=sentences
                )
                
                track = hybrid_result["track"]
                track_name = hybrid_result["track_name"]
                is_hit = hybrid_result["is_covered"]
                final_score = hybrid_result["final_score"]
                disagreement = hybrid_result["disagreement"]
                needs_llm = hybrid_result["needs_llm"]
                
                # Track 1: FAST - instant decision
                if track == 1:
                    verification_method = f"hybrid-{track_name}"
                    print(f"  ⚡ '{key_point_text[:30]}...': Track 1 ({track_name}) sim={similarity:.3f} → {'PASS' if is_hit else 'FAIL'}")
                
                # Track 2: STANDARD - NLI verified
                elif track == 2:
                    verification_method = f"hybrid-{track_name}"
                    print(f"  ✅ '{key_point_text[:30]}...': Track 2 (standard) sim={similarity:.3f}→{final_score:.3f}, disagree={disagreement:.3f}")
                
                # Track 3: DEEP - Multi-model verification
                elif track == 3:
                    verification_method = f"hybrid-{track_name}"
                    print(f"  🔍 '{key_point_text[:30]}...': Track 3 (deep) sim={similarity:.3f}→{final_score:.3f}, disagree={disagreement:.3f}")
                
                # Track 4: CRITICAL - Needs LLM arbiter
                elif track == 4:
                    verification_method = f"hybrid-{track_name}"
                    nli_scores = hybrid_result.get("nli_scores", {})
                    nli_small = nli_scores.get("small", {})
                    
                    # Check if LLM arbiter should be triggered
                    if needs_llm and self._llm_arbiter:
                        # Calculate uncertainty for LLM trigger
                        uncertainty = self._calculate_uncertainty_score(
                            similarity=similarity,
                            nli_entailment=nli_small.get("best_entailment", 0.5),
                            nli_contradiction=nli_small.get("max_contradiction", 0),
                            overlap=overlap
                        )
                        
                        if self._llm_arbiter.should_trigger(uncertainty):
                            print(f"     🤖 Track 4: High disagreement ({disagreement:.2f}) + uncertainty ({uncertainty:.2f}) - triggering LLM arbiter")
                            
                            import asyncio
                            try:
                                llm_result = asyncio.run(self._llm_arbiter.verify_answer(
                                    question_text=question.get("question_text", ""),
                                    key_point=key_point_text,
                                    student_answer=user_answer,
                                    cosine_similarity=similarity,
                                    nli_entailment=nli_small.get("best_entailment", 0.5),
                                    nli_contradiction=nli_small.get("has_contradiction", False)
                                ))
                                
                                # LLM overrides the decision
                                is_hit = llm_result["is_correct"]
                                verification_method = "hybrid-llm-arbiter"
                                print(f"     ✨ LLM decision: {'PASS' if is_hit else 'FAIL'} (confidence: {llm_result['confidence']:.2f})")
                                print(f"        Reasoning: {llm_result['reasoning']}")
                            except Exception as e:
                                print(f"     ⚠️ LLM arbiter failed: {e}, using NLI decision")
                    
                    print(f"  ⚠️ '{key_point_text[:30]}...': Track 4 (critical) sim={similarity:.3f}→{final_score:.3f}, disagree={disagreement:.3f}")
                
                # Log uncertain cases for active learning
                if hybrid_result.get("confidence", 1.0) < 0.7 or needs_llm:
                    uncertainty = self._calculate_uncertainty_score(
                        similarity=similarity,
                        nli_entailment=hybrid_result.get("nli_scores", {}).get("small", {}).get("best_entailment", 0),
                        nli_contradiction=hybrid_result.get("nli_scores", {}).get("small", {}).get("max_contradiction", 0),
                        overlap=overlap
                    )
                    
                    if uncertainty >= self._uncertainty_threshold:
                        self._log_uncertain_case(
                            question_id=question_id,
                            question_text=question.get("question_text", ""),
                            key_point=key_point_text,
                            student_answer=user_answer,
                            similarity=similarity,
                            nli_result=hybrid_result.get("nli_scores", {}).get("small", {}),
                            overlap=overlap,
                            decision=is_hit,
                            uncertainty_score=uncertainty
                        )
            
            else:
                # Fallback to pure embedding mode
                is_hit = self._is_key_point_hit(similarity, overlap)
                verification_method = "embedding-only"
                print(f"  📊 '{key_point_text[:30]}...': sim={similarity:.3f}, overlap={overlap:.2f}, hit={is_hit}")
            
            # Track confidence metrics for smart LLM fallback
            had_contradiction = False
            if hybrid_result:
                nli_scores = hybrid_result.get("nli_scores", {})
                if nli_scores:
                    small_scores = nli_scores.get("small", {})
                    if small_scores and small_scores.get("has_contradiction", False):
                        had_contradiction = True
            
            key_point_confidence = {
                "cosine": similarity,
                "is_hit": is_hit,
                "in_gray_zone": 0.58 <= similarity <= 0.78,  # Uncertain range
                "had_contradiction": had_contradiction
            }
            confidence_scores.append(key_point_confidence)
            
            # Categorize result
            if is_hit:
                hit_key_points.append(key_point_text)
            else:
                missing_key_points.append(key_point_text)
        
        # ==========================================
        # LLM SMART FALLBACK - Confidence-Based
        # ===========================================
        # Only call LLM when there's genuine uncertainty
        llm_holistic_mode = getattr(settings.grading, 'llm_holistic_mode', 'never')
        
        if llm_holistic_mode != "never" and self._llm_arbiter:
            should_use_llm = False
            llm_reason = ""
            
            if llm_holistic_mode == "always":
                should_use_llm = True
                llm_reason = "mode=always"
            
            elif llm_holistic_mode == "fallback":
                preliminary_score = self._calculate_score(len(hit_key_points), len(key_points))
                
                # Calculate overall confidence
                gray_zone_count = sum(1 for c in confidence_scores if c["in_gray_zone"])
                contradiction_count = sum(1 for c in confidence_scores if c["had_contradiction"])
                
                # HIGH CONFIDENCE (skip LLM):
                # - Score is 0% AND all cosine < 0.55 (clearly wrong)
                # - Score is 100% AND all cosine > 0.80 (clearly correct)
                all_low = all(c["cosine"] < 0.55 for c in confidence_scores)
                all_high = all(c["cosine"] > 0.80 for c in confidence_scores)
                
                if preliminary_score == 0 and all_low:
                    should_use_llm = False
                    print(f"\n✅ High confidence: 0% with all low cosine - skipping LLM")
                elif preliminary_score == 100 and all_high:
                    should_use_llm = False
                    print(f"\n✅ High confidence: 100% with all high cosine - skipping LLM")
                # LOW CONFIDENCE (use LLM):
                # - Score in middle range (25-75%)
                # - Any key point in gray zone
                # - Any contradiction detected
                elif 25 <= preliminary_score <= 75:
                    should_use_llm = True
                    llm_reason = f"uncertain score ({preliminary_score:.0f}%)"
                elif gray_zone_count > 0:
                    should_use_llm = True
                    llm_reason = f"{gray_zone_count} key points in gray zone"
                elif contradiction_count > 0:
                    should_use_llm = True
                    llm_reason = f"{contradiction_count} contradictions detected"
                # MEDIUM CONFIDENCE: 0-25% or 75-100% with clear signals
                else:
                    should_use_llm = False
                    print(f"\n✅ Medium confidence: {preliminary_score:.0f}% with clear signals - skipping LLM")
            
            if should_use_llm:
                print(f"\n🤖 LLM Fallback triggered: {llm_reason}")
                import asyncio
                try:
                    key_point_texts = [kp["text"] for kp in key_points]
                    llm_result = asyncio.run(self._llm_arbiter.grade_holistically(
                        question_text=question.get("question_text", ""),
                        key_points=key_point_texts,
                        student_answer=user_answer
                    ))
                    
                    if llm_result:
                        # Override with LLM's holistic assessment
                        hit_key_points = []
                        missing_key_points = []
                        
                        for i, (kp_text, is_covered) in enumerate(zip(key_point_texts, llm_result["covered"])):
                            if is_covered:
                                hit_key_points.append(kp_text)
                            else:
                                missing_key_points.append(kp_text)
                        
                        print(f"   ✨ LLM Override: {len(hit_key_points)}/{len(key_points)} key points covered")
                        
                except Exception as e:
                    print(f"   ⚠️ LLM fallback failed: {e}, using embedding/NLI results")
        
        # Calculate score and generate feedback
        score = self._calculate_score(len(hit_key_points), len(key_points))
        feedback = self._generate_feedback(score, len(missing_key_points))
        
        # ==========================================
        # CACHE THE GRADED ANSWER
        # ==========================================
        self._answer_cache_service.cache_graded_answer(
            question_id=question_id,
            student_answer=user_answer,
            score=score,
            hit_key_points=hit_key_points,
            missing_key_points=missing_key_points,
            feedback=feedback
        )
        
        return AnswerResponse(
            score=round(score, 1),
            hit_key_points=hit_key_points,
            missing_key_points=missing_key_points,
            feedback=feedback
        )
    
    def _apply_llm_verification(
        self,
        similarity: float,
        best_sentence: str,
        reference_chunks: List[str],
    ) -> Optional[LLMVerificationResult]:
        """
        Optionally adjust similarity using LLM verification for borderline cases.
        """
        if not self._llm_verification or not best_sentence.strip():
            return None

        return self._llm_verification.verify_chunk(
            best_sentence,
            reference_chunks,
            similarity,
        )
    
    def _detect_antonym_conflict(self, user_answer: str, key_point_text: str, question_text: str) -> bool:
        """
        Detect if user answer and key point are antonyms using AI-powered detection
        
        Args:
            user_answer: User's answer text
            key_point_text: Key point text to compare against
            question_text: Question context for better detection
            
        Returns:
            True if antonym conflict is detected with sufficient confidence
        """
        try:
            # Use AI antonym detector to analyze the relationship
            result = self._ai_antonym_detector.detect_antonyms(
                user_answer, key_point_text, question_text
            )
            
            # Only consider it a conflict if we have sufficient confidence
            min_confidence = self._antonym_config.min_confidence_for_penalty
            confidence_thresholds = {
                "high": AntonymConfidence.HIGH,
                "medium": AntonymConfidence.MEDIUM,
                "low": AntonymConfidence.LOW
            }
            
            required_confidence = confidence_thresholds.get(min_confidence, AntonymConfidence.MEDIUM)
            
            # Check if we have sufficient confidence and detected antonym relationship
            confidence_levels = [AntonymConfidence.HIGH, AntonymConfidence.MEDIUM, AntonymConfidence.LOW, AntonymConfidence.NONE]
            required_index = confidence_levels.index(required_confidence)
            result_index = confidence_levels.index(result.confidence)
            
            has_sufficient_confidence = result_index <= required_index
            is_antonym = result.is_antonym
            
            if is_antonym and has_sufficient_confidence:
                print(f"  🚫 Antonym conflict detected: '{user_answer[:30]}...' vs '{key_point_text[:30]}...' "
                      f"(confidence: {result.confidence.value}, method: {result.method})")
                return True
            
            return False
            
        except Exception as e:
            print(f"  ⚠️ Error in antonym detection: {e}")
            # Fallback to original method if AI detection fails
            user_tokens = self._text_processor.normalize_text(user_answer)
            key_point_tokens = self._text_processor.normalize_text(key_point_text)
            return (self._text_processor.has_polarity_conflict(user_tokens, key_point_tokens) or
                    self._text_processor.has_direction_conflict(user_tokens, key_point_tokens))
    
    def _is_key_point_hit(self, similarity: float, overlap: float) -> bool:
        """
        Determine if a key point is hit based on similarity and overlap thresholds
        
        Args:
            similarity: Semantic similarity score
            overlap: Lexical overlap score
            
        Returns:
            True if key point is considered hit, False otherwise
        """
        return (similarity >= self._similarity_config.high_similarity or 
                (similarity >= self._similarity_config.mid_similarity and 
                 overlap >= self._similarity_config.min_lexical_overlap))
    
    def _compute_embeddings_for_question(self, question: Dict) -> None:
        """
        Compute embeddings for a single question's key points and store them in memory.

        Args:
            question: Question dictionary containing key_points
        """
        question_id = question["question_id"]
        embeddings = []
        keywords_list: List[Set[str]] = []

        for key_point in question["key_points"]:
            text = key_point["text"]
            embedding = self._embedding_service.get_embedding(text)
            embeddings.append(embedding)
            keywords = set(self._text_processor.normalize_text(text))
            keywords_list.append(keywords)
            print(f"  ✅ (on-demand) Embedded: '{text[:30]}...'")

        self._key_point_embeddings[question_id] = embeddings
        self._key_point_keywords[question_id] = keywords_list
        print(f"  📝 (on-demand) Question {question_id}: {len(embeddings)} key points embedded")

    def _find_best_sentence_match(
        self,
        sentences: List[str],
        sentence_embeddings: List[List[float]],
        key_point_embedding: List[float],
    ) -> Tuple[float, str]:
        """
        Find the best matching sentence and its similarity score for a key point.
        """
        if not sentence_embeddings:
            return 0.0, ""

        best_similarity = -1.0
        best_sentence = ""

        for idx, embedding in enumerate(sentence_embeddings):
            similarity = self._embedding_service.compute_cosine_similarity(
                embedding, key_point_embedding
            )
            if similarity > best_similarity:
                best_similarity = similarity
                if idx < len(sentences):
                    best_sentence = sentences[idx]

        if best_similarity < 0:
            return 0.0, best_sentence

        return best_similarity, best_sentence
    
    def _calculate_score(self, hit_count: int, total_count: int) -> float:
        """
        Calculate percentage score based on hit key points
        
        Args:
            hit_count: Number of key points hit
            total_count: Total number of key points
            
        Returns:
            Score as percentage (0-100)
        """
        if total_count == 0:
            return 0.0
        
        return (hit_count / total_count) * 100
    
    def _generate_feedback(self, score: float, missing_count: int) -> str:
        """
        Generate feedback message based on score and missing key points
        
        Args:
            score: Calculated score percentage
            missing_count: Number of missing key points
            
        Returns:
            Appropriate feedback message
        """
        if score == 100:
            return self._feedback_config.perfect_score
        elif score >= 50:
            return self._feedback_config.partial_score.format(
                missing_count=missing_count
            )
        else:
            return self._feedback_config.low_score
    
    def get_answer_cache_stats(self, question_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get statistics about cached answers
        
        Args:
            question_id: Optional question ID to filter stats
            
        Returns:
            Dictionary with cache statistics
        """
        return self._answer_cache_service.get_cache_stats(question_id)
    
    def clear_answer_cache_for_question(self, question_id: int) -> bool:
        """
        Clear cached answers for a specific question
        
        Useful when question content changes or corrections are needed.
        
        Args:
            question_id: ID of the question to clear
            
        Returns:
            True if cleared successfully
        """
        return self._answer_cache_service.clear_cache_for_question(question_id)
