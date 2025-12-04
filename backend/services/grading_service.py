"""
Answer grading service using embeddings and text analysis
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from fastapi import HTTPException

from models.models import AnswerResponse
from services.embedding_service import EmbeddingService
from services.text_processing import TextProcessor
from services.question_service import QuestionService
from services.embedding_storage import EmbeddingStorage
from services.smart_antonym_detector import SmartAntonymDetector, AntonymConfidence
from services.verification import LLMVerificationService, LLMVerificationResult
from core.config import settings


class GradingService:
    """
    Service for grading user answers against question key points
    
    This service handles:
    - Precomputing embeddings for key points
    - Grading user answers using semantic similarity
    - Combining semantic and lexical analysis
    - Generating feedback based on grading results
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
        
        # Storage for precomputed embeddings and keywords
        self._key_point_embeddings: Dict[int, List[List[float]]] = {}
        self._key_point_keywords: Dict[int, List[Set[str]]] = {}
        
        # Configuration
        self._similarity_config = settings.grading.similarity_thresholds
        self._feedback_config = settings.grading.feedback_messages
        self._validation_config = settings.grading.answer_validation
        self._antonym_config = settings.antonym_detection
    
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
        
        # Evaluate each key point
        hit_key_points = []
        missing_key_points = []
        
        reference_chunks = [kp["text"] for kp in key_points]

        for i, key_point in enumerate(key_points):
            key_point_embedding = self._key_point_embeddings[question_id][i]
            key_point_tokens = list(self._key_point_keywords[question_id][i])
            
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
            
            # Check for semantic conflicts using AI-powered antonym detection
            has_conflict = self._detect_antonym_conflict(user_answer, key_point["text"], question["question_text"])
            
            # If we detect opposing claims, apply penalty based on confidence
            if has_conflict:
                penalty_multiplier = self._antonym_config.antonym_penalty_multiplier
                similarity *= penalty_multiplier
                overlap *= penalty_multiplier
            penalized_similarity = similarity
            
            llm_result = self._apply_llm_verification(
                similarity, best_sentence, reference_chunks
            )
            llm_confirmed = False
            llm_executed = False
            if llm_result:
                llm_executed = llm_result.llm_used
                if llm_result.llm_used and llm_result.decision:
                    llm_confirmed = llm_result.decision.is_similar
                    if (
                        self._similarity_config.llm_verification_threshold
                        < raw_similarity
                        < self._similarity_config.high_similarity
                    ):
                        status = "PASS" if llm_confirmed else "FAIL"
                        print(
                            f"    🤖 LLM verification ({status}): reason={llm_result.reason}"
                        )

            # Determine if key point is hit based on adjusted scores
            is_hit = (not has_conflict) and self._is_key_point_hit(similarity, overlap)
            if not is_hit and llm_confirmed:
                is_hit = True
            
            if is_hit:
                hit_key_points.append(key_point["text"])
            else:
                missing_key_points.append(key_point["text"])
            
            print(
                f"  📊 Key point '{key_point['text'][:30]}...': "
                f"sim_raw={raw_similarity:.3f}, "
                f"sim_penalized={penalized_similarity:.3f}, "
                f"sim_final={similarity:.3f}, "
                f"overlap_raw={raw_overlap:.2f}, "
                f"overlap_final={overlap:.2f}, "
                f"hit={is_hit}"
                + (", llm=used" if llm_executed else "")
            )
        
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
