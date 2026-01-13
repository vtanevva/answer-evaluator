"""
Answer grading service using embeddings and text analysis
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from fastapi import HTTPException
import json
import asyncio
import nest_asyncio
from datetime import datetime
from pathlib import Path

from models.models import AnswerResponse
from services.embedding_service import EmbeddingService
from services.text_processing import TextProcessor
from services.question_service import QuestionService
from services.embedding_storage import EmbeddingStorage
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
    
    def _call_llm_sync(self, question_text: str, key_point: str, student_answer: str, cosine_similarity: float) -> dict:
        """
        Call LLM arbiter synchronously (wrapper for async method)
        Uses asyncio.get_event_loop() to run in existing event loop
        """
        # Allow nested event loops (for FastAPI compatibility)
        nest_asyncio.apply()
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async LLM call
        return loop.run_until_complete(
            self._llm_arbiter.verify_answer(
                question_text=question_text,
                key_point=key_point,
                student_answer=student_answer,
                cosine_similarity=cosine_similarity,
                nli_entailment=0.0,
                nli_contradiction=False
            )
        )
    
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
        1. Validate the answer
        2. Get user answer embeddings (sentence-level)
        3. Compare with each key point embedding using cosine similarity
        4. Mark key points as "hit" or "missing" based on similarity threshold
        5. Calculate score and generate feedback
        
        Args:
            question_id: ID of the question being answered
            user_answer: The student's answer text
            
        Returns:
            AnswerResponse with score, hit/missing points, and feedback
        """
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
        
        # GLOBAL CONTRADICTION CHECK: DISABLED (too aggressive, causes false negatives)
        # This was causing short but correct answers to fail completely
        global_contradiction_detected = False
        # if getattr(settings.grading, 'grading_method', 'embedding') == "hybrid" and self._nli_service:
        #     # Check the main question text against the answer for fundamental contradictions
        #     question_text = question["question_text"]
        #     global_nli_result = self._nli_service.evaluate_answer_against_keypoint(
        #         student_answer=user_answer,
        #         key_point=question_text,
        #         sentences=sentences
        #     )
        #     # Only flag global contradiction if NOT a comprehensive answer
        #     # (comprehensive answers naturally discuss negatives, which shouldn't be flagged)
        #     if not is_comprehensive_answer and global_nli_result["has_contradiction"] and global_nli_result["best_entailment"] < 0.3:
        #         global_contradiction_detected = True
        #         print(f"\n⚠️ GLOBAL CONTRADICTION: Answer contradicts the question premise")
        #         print(f"   Question: '{question_text}'")
        #         print(f"   Contradiction score: {global_nli_result['has_contradiction']}")
        
        # ==========================================
        # LLM-FIRST MODE: Skip NLI/embedding when "always"
        # ==========================================
        llm_holistic_mode = getattr(settings.grading, 'llm_holistic_mode', 'never')
        
        print(f"\n🔍 DEBUG: llm_holistic_mode='{llm_holistic_mode}', llm_arbiter={self._llm_arbiter is not None}")
        
        if llm_holistic_mode == "always" and self._llm_arbiter:
            print(f"\n🤖 LLM-FIRST MODE: Using LLM for all grading (mode=always)")
            import asyncio
            try:
                key_point_texts = [kp["text"] for kp in key_points]
                llm_result = asyncio.run(self._llm_arbiter.grade_holistically(
                    question_text=question.get("question_text", ""),
                    key_points=key_point_texts,
                    student_answer=user_answer,
                    sentences=sentences
                ))
                
                if llm_result:
                    hit_key_points = []
                    missing_key_points = []
                    
                    for kp_text, is_covered in zip(key_point_texts, llm_result["covered"]):
                        if is_covered:
                            hit_key_points.append(kp_text)
                        else:
                            missing_key_points.append(kp_text)
                    
                    score = self._calculate_score(len(hit_key_points), len(key_points))
                    feedback = self._generate_feedback(score, len(missing_key_points))
                    
                    print(f"   ✨ LLM Result: {len(hit_key_points)}/{len(key_points)} key points covered ({score:.1f}%)")
                    
                    return AnswerResponse(
                        score=round(score, 1),
                        hit_key_points=hit_key_points,
                        missing_key_points=missing_key_points,
                        feedback=feedback
                    )
                else:
                    print(f"   ⚠️ LLM returned None, falling back to NLI/embedding")
                    
            except Exception as e:
                import traceback
                print(f"   ⚠️ LLM-first failed: {e}")
                print(f"   {traceback.format_exc()}")
                print(f"   Falling back to NLI/embedding grading")
        
        # Evaluate each key point (NLI/embedding fallback)
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
            
            elif getattr(settings.grading, 'grading_method', 'embedding') == "nli" and self._nli_service:
                # ===========================================
                # NLI-FIRST ARCHITECTURE (No Cosine Similarity)
                # ===========================================
                # NLI is the primary grader - better at paraphrasing and contradictions
                # LLM is called only for uncertain cases (neutral zone)
                # ===========================================
                
                # Use NLI with full answer context
                nli_result = self._nli_service._evaluate_sentences_nli(
                    sentences=sentences,
                    key_point=key_point_text,
                    use_base=True,  # Use base model for better accuracy
                    full_answer=user_answer
                )
                
                entailment = nli_result["best_entailment"]
                full_entailment = nli_result.get("full_answer_entailment", 0)
                has_contradiction = nli_result["has_contradiction"]
                entailment_source = nli_result.get("entailment_source", "sentence")
                
                # Get thresholds from settings
                entail_threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.65)
                contradict_threshold = getattr(settings.grading, 'nli_contradiction_threshold', 0.70)
                uncertain_low = getattr(settings.grading, 'nli_uncertain_low', 0.40)
                uncertain_high = getattr(settings.grading, 'nli_uncertain_high', 0.65)
                
                # Decision logic
                needs_llm = False
                
                # CASE 1: CONTRADICTION DETECTED - Fail immediately
                if has_contradiction:
                    is_hit = False
                    verification_method = "nli-contradiction"
                    print(f"  ❌ '{key_point_text[:30]}...': NLI CONTRADICTION detected → FAIL")
                
                # CASE 2: HIGH ENTAILMENT - Key point clearly covered
                elif entailment >= entail_threshold:
                    is_hit = True
                    verification_method = f"nli-entailed-{entailment_source}"
                    print(f"  ✅ '{key_point_text[:30]}...': NLI entailment={entailment:.3f} (via {entailment_source}) → PASS")
                
                # CASE 3: LOW ENTAILMENT - Key point clearly NOT covered
                elif entailment < uncertain_low:
                    is_hit = False
                    verification_method = "nli-not-entailed"
                    print(f"  ❌ '{key_point_text[:30]}...': NLI entailment={entailment:.3f} < {uncertain_low} → FAIL")
                
                # CASE 4: UNCERTAIN ZONE - Need LLM verification
                else:
                    needs_llm = True
                    # Tentative decision based on above/below midpoint
                    midpoint = (uncertain_low + uncertain_high) / 2
                    tentative_hit = entailment >= midpoint
                    
                    print(f"  🔄 '{key_point_text[:30]}...': NLI uncertain zone ({uncertain_low} ≤ {entailment:.3f} < {uncertain_high}) → LLM needed")
                    
                    # Try LLM for this key point if available
                    if self._llm_arbiter:
                        import asyncio
                        try:
                            llm_result = asyncio.run(self._llm_arbiter.verify_answer(
                                question_text=question.get("question_text", ""),
                                key_point=key_point_text,
                                student_answer=user_answer,
                                cosine_similarity=0.0,  # Not using cosine
                                nli_entailment=entailment,
                                nli_contradiction=has_contradiction
                            ))
                            
                            is_hit = llm_result["is_correct"]
                            verification_method = "nli-llm-verified"
                            print(f"     🤖 LLM decision: {'PASS' if is_hit else 'FAIL'} (confidence: {llm_result['confidence']:.2f})")
                            print(f"        Reasoning: {llm_result['reasoning']}")
                        except Exception as e:
                            print(f"     ⚠️ LLM failed: {e}, using tentative NLI decision")
                            is_hit = tentative_hit
                            verification_method = "nli-uncertain-fallback"
                    else:
                        is_hit = tentative_hit
                        verification_method = "nli-uncertain-no-llm"
                
                # Store hybrid_result for consistency with rest of code
                hybrid_result = {
                    "track": 2 if is_hit else 1,
                    "track_name": "nli-first",
                    "is_covered": is_hit,
                    "final_score": entailment,
                    "confidence": 1.0 - abs(entailment - 0.5) * 2,  # Confidence based on distance from midpoint
                    "nli_scores": {"small": nli_result},
                    "disagreement": 0.0,
                    "needs_llm": needs_llm
                }
            
            elif getattr(settings.grading, 'grading_method', 'embedding') == "hybrid" and self._nli_service:
                # ===========================================
                # OPTIMIZED TIERED COSINE-NLI-LLM ARCHITECTURE
                # ===========================================
                # Tier 1: Cosine ≥ 92%  → Auto-PASS (obvious match)
                # Tier 2: Cosine 75-92% → LLM Arbiter (gray zone reasoning)
                # Tier 3: Cosine 60-75% → NLI Check (contradiction detection)
                # Tier 4: Cosine < 60%  → Auto-FAIL (clearly not matching)
                # ===========================================
                
                # Get tier thresholds from settings
                tier1_threshold = getattr(settings.grading, 'tier1_auto_pass', 0.92)
                tier2_threshold = getattr(settings.grading, 'tier2_llm_min', 0.75)
                tier3_threshold = getattr(settings.grading, 'tier3_nli_min', 0.60)
                
                hybrid_result = None
                
                # TIER 1: AUTO-PASS (cosine ≥ 95%)
                if similarity >= tier1_threshold:
                    is_hit = True
                    verification_method = "tier1-auto-pass"
                    print(f"  ✅ '{key_point_text[:30]}...': TIER 1 (auto-pass) cosine={similarity:.3f} ≥ {tier1_threshold} → PASS")
                
                # TIER 2: LLM ARBITER (cosine 75-92%)
                elif similarity >= tier2_threshold:
                    verification_method = "tier2-llm"
                    print(f"  🤖 '{key_point_text[:30]}...': TIER 2 (LLM) cosine={similarity:.3f} in [{tier2_threshold}, {tier1_threshold})")
                    
                    if self._llm_arbiter:
                        try:
                            # Call LLM synchronously (avoid asyncio.run in async context)
                            llm_result = self._call_llm_sync(
                                question_text=question.get("question_text", ""),
                                key_point=key_point_text,
                                student_answer=user_answer,
                                cosine_similarity=similarity
                            )
                            
                            is_hit = llm_result["is_correct"]
                            print(f"     ✨ LLM decision: {'PASS' if is_hit else 'FAIL'} (confidence: {llm_result['confidence']:.2f})")
                            print(f"        Reasoning: {llm_result['reasoning']}")
                        except Exception as e:
                            print(f"     ⚠️ LLM failed: {e}, defaulting to PASS (high cosine)")
                            is_hit = True  # Default to pass since cosine is high
                    else:
                        # No LLM available, optimistically pass since cosine is decent
                        is_hit = True
                        print(f"     ⚠️ No LLM arbiter, defaulting to PASS")
                
                # TIER 3: NLI CHECK (cosine 50-75%)
                elif similarity >= tier3_threshold:
                    verification_method = "tier3-nli"
                    print(f"  🔍 '{key_point_text[:30]}...': TIER 3 (NLI) cosine={similarity:.3f} in [{tier3_threshold}, {tier2_threshold})")
                    
                    # Use NLI to verify
                    nli_result = self._nli_service._evaluate_sentences_nli(
                        sentences=sentences,
                        key_point=key_point_text,
                        use_base=True,
                        full_answer=user_answer
                    )
                    
                    entailment = nli_result["best_entailment"]
                    has_contradiction = nli_result["has_contradiction"]
                    entail_threshold = getattr(settings.grading, 'nli_entailment_threshold', 0.60)
                    
                    if has_contradiction:
                        is_hit = False
                        print(f"     ❌ NLI: CONTRADICTION detected → FAIL")
                    elif entailment >= entail_threshold:
                        is_hit = True
                        print(f"     ✅ NLI: entailment={entailment:.3f} ≥ {entail_threshold} → PASS")
                    else:
                        is_hit = False
                        print(f"     ❌ NLI: entailment={entailment:.3f} < {entail_threshold} → FAIL")
                    
                    hybrid_result = {
                        "nli_scores": {"small": nli_result},
                        "final_score": entailment
                    }
                
                # TIER 4: AUTO-FAIL (cosine < 50%)
                else:
                    is_hit = False
                    verification_method = "tier4-auto-fail"
                    print(f"  ❌ '{key_point_text[:30]}...': TIER 4 (auto-fail) cosine={similarity:.3f} < {tier3_threshold} → FAIL")
                
                # Log uncertain cases for active learning (only for Tier 3 with NLI)
                if hybrid_result and hybrid_result.get("nli_scores"):
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
        
        print(f"\n🔍 DEBUG: llm_holistic_mode={llm_holistic_mode}, llm_arbiter={self._llm_arbiter is not None}")
        
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
                print(f"   Calling grade_holistically with {len(key_points)} key points...")
                import asyncio
                try:
                    key_point_texts = [kp["text"] for kp in key_points]
                    llm_result = asyncio.run(self._llm_arbiter.grade_holistically(
                        question_text=question.get("question_text", ""),
                        key_points=key_point_texts,
                        student_answer=user_answer,
                        sentences=sentences  # NEW: Pass sentences for full-context LLM analysis
                    ))
                    
                    print(f"   LLM result received: {llm_result is not None}")
                    
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
                    else:
                        print(f"   ⚠️ LLM returned None, using NLI results")
                        
                except Exception as e:
                    import traceback
                    print(f"   ⚠️ LLM fallback failed: {e}")
                    print(f"   Traceback: {traceback.format_exc()}")
        
        # Calculate score and generate feedback
        score = self._calculate_score(len(hit_key_points), len(key_points))
        feedback = self._generate_feedback(score, len(missing_key_points))
        
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
