"""
NLI-Enhanced Grading Service

This service combines:
1. NLI (Natural Language Inference) for semantic understanding
2. Optional embedding similarity as a backup
3. Automatic handling of negations, antonyms, paraphrasing

Benefits over pure embedding similarity:
- Understands "not", "opposite", "contra-" patterns
- Detects contradictions (e.g., "increasing" vs "decreasing")
- Better with paraphrasing and synonyms
- No need for separate antonym/negation filters
"""

from typing import List, Dict, Optional
from fastapi import HTTPException

from models.models import AnswerResponse
from services.nli_service import NLIService
from services.text_processing import TextProcessor
from services.question_service import QuestionService
from core.config import settings


class NLIGradingService:
    """
    Enhanced grading service using Natural Language Inference
    
    This replaces cosine similarity with NLI for more robust grading that:
    - Understands semantic meaning and gist
    - Handles negations and antonyms automatically
    - Works with paraphrasing without word matching
    - Provides explainable confidence scores
    """
    
    def __init__(self, question_service: QuestionService):
        """
        Initialize NLI grading service
        
        Args:
            question_service: Service for managing questions
        """
        self._question_service = question_service
        self._nli_service = NLIService()
        self._text_processor = TextProcessor()
        
        # Configuration
        self._feedback_config = settings.grading.feedback_messages
        self._validation_config = settings.grading.answer_validation
        
        # NLI thresholds (configurable in settings.yaml)
        self._entailment_threshold = getattr(
            settings.grading, 'nli_entailment_threshold', 0.6
        )
        self._contradiction_threshold = getattr(
            settings.grading, 'nli_contradiction_threshold', 0.7
        )
        
        print("✅ NLI Grading Service initialized")
        print(f"   Entailment threshold: {self._entailment_threshold}")
        print(f"   Contradiction threshold: {self._contradiction_threshold}")
    
    def validate_answer(self, user_answer: str) -> Optional[AnswerResponse]:
        """
        Validate user answer for basic requirements
        
        Args:
            user_answer: The user's answer text
            
        Returns:
            AnswerResponse with error feedback if invalid, None if valid
        """
        user_answer_clean = user_answer.strip()
        
        # Check for empty or "I don't know" answers
        if (not user_answer_clean or 
            user_answer_clean.lower() in self._validation_config.invalid_answers):
            return AnswerResponse(
                score=0.0,
                hit_key_points=[],
                missing_key_points=[],
                feedback=self._feedback_config.empty_answer
            )
        
        # Check for very short answers
        if (len(user_answer_clean) < self._validation_config.min_answer_length or 
            len(user_answer_clean.split()) < self._validation_config.min_word_count):
            return AnswerResponse(
                score=0.0,
                hit_key_points=[],
                missing_key_points=[],
                feedback=self._feedback_config.short_answer
            )
        
        return None  # Answer is valid
    
    def grade_answer(self, question_id: int, user_answer: str) -> AnswerResponse:
        """
        Grade user answer using NLI-based semantic analysis
        
        Process:
        1. Validate answer
        2. Split into sentences for fine-grained analysis
        3. Use NLI to check each key point:
           - Entailment: student answer supports key point ✓
           - Contradiction: student answer contradicts key point ✗
           - Neutral: unclear relationship
        4. Calculate score and generate feedback
        
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
        
        # Split user answer into sentences for better NLI analysis
        sentences = self._text_processor.split_into_sentences(user_answer)
        
        # Evaluate each key point using NLI
        hit_key_points = []
        missing_key_points = []
        key_point_details = []
        total_weighted_score = 0.0
        total_weight = 0.0
        has_any_contradiction = False  # Track if answer contradicts ANY key point
        
        print(f"\n🔍 Grading answer for question {question_id}")
        print(f"   Student answer: {user_answer[:100]}...")
        
        for key_point in key_points:
            key_point_text = key_point["text"]
            key_point_weight = key_point.get("weight", 1.0)  # Support weighted questions
            
            # Use NLI to evaluate this key point
            nli_result = self._nli_service.evaluate_answer_against_keypoint(
                student_answer=user_answer,
                key_point=key_point_text,
                sentences=sentences
            )
            
            confidence = nli_result["confidence"]
            has_contradiction = nli_result["has_contradiction"]
            best_entailment = nli_result["best_entailment"]
            
            # Track global contradiction
            if has_contradiction:
                has_any_contradiction = True
            
            # WEIGHTED SCORING: Use actual NLI confidence instead of binary
            # Apply contradiction penalty
            if has_contradiction:
                key_point_score = 0.0  # Strong contradiction = 0 points
            else:
                # Use best_entailment as the score for this key point
                # Scale from [0, 1] to [0, 100] percentage
                key_point_score = best_entailment * 100
            
            # Apply weight
            weighted_contribution = key_point_score * key_point_weight
            total_weighted_score += weighted_contribution
            total_weight += key_point_weight
            
            # Determine if "covered" for feedback purposes (>60% threshold)
            is_covered = best_entailment >= self._entailment_threshold and not has_contradiction
            
            # Log detailed results
            print(f"\n   📊 Key Point: '{key_point_text}'")
            print(f"      Entailment: {best_entailment:.3f}")
            print(f"      Key Point Score: {key_point_score:.1f}%")
            print(f"      Contradiction: {has_contradiction}")
            print(f"      Covered: {is_covered}")
            
            # Store for potential feedback
            key_point_details.append({
                "text": key_point_text,
                "covered": is_covered,
                "confidence": confidence,
                "entailment": best_entailment,
                "key_point_score": key_point_score,
                "contradiction": has_contradiction
            })
            
            # Categorize as hit or missing for feedback
            if is_covered:
                hit_key_points.append(key_point_text)
            else:
                missing_key_points.append(key_point_text)
        
        # Calculate final score as weighted average
        if total_weight > 0:
            score = total_weighted_score / total_weight
        else:
            score = 0.0
        
        # CRITICAL: If answer contradicts ANY key point, cap score at 0%
        # This prevents false positives from inflating contradictory answers
        if has_any_contradiction:
            contradiction_count = sum(1 for kp in key_point_details if kp["contradiction"])
            print(f"\n   ⚠️ CONTRADICTION DETECTED: {contradiction_count} key point(s) contradicted")
            print(f"   ⚠️ Applying contradiction penalty: Score capped at 0%")
            score = 0.0
        
        # Generate feedback
        feedback = self._generate_feedback(
            score, 
            len(missing_key_points),
            key_point_details
        )
        
        print(f"\n   ✅ Final Score: {score}%")
        print(f"      Hit: {len(hit_key_points)}/{len(key_points)}")
        
        return AnswerResponse(
            score=round(score, 1),
            hit_key_points=hit_key_points,
            missing_key_points=missing_key_points,
            feedback=feedback
        )
    
    def _calculate_score(self, hit_count: int, total_count: int) -> float:
        """
        DEPRECATED: Kept for backward compatibility
        Now using weighted NLI scoring in grade_answer() directly
        
        Args:
            hit_count: Number of key points covered
            total_count: Total number of key points
            
        Returns:
            Score as percentage (0-100)
        """
        if total_count == 0:
            return 0.0
        
        return (hit_count / total_count) * 100
    
    def _generate_feedback(
        self, 
        score: float, 
        missing_count: int,
        key_point_details: List[Dict]
    ) -> str:
        """
        Generate detailed feedback based on NLI analysis
        
        Args:
            score: Calculated score percentage
            missing_count: Number of missing key points
            key_point_details: Detailed NLI results per key point
            
        Returns:
            Feedback message
        """
        # Check for contradictions
        contradictions = [
            kp for kp in key_point_details 
            if kp.get("contradiction", False)
        ]
        
        if contradictions and score < 100:
            # Special feedback for contradictory answers
            return (
                f"Your answer contains contradictions to key points. "
                f"Score: {score:.0f}%. Review the material carefully."
            )
        
        # Standard feedback
        if score == 100:
            return self._feedback_config.perfect_score
        elif score >= 50:
            return self._feedback_config.partial_score.format(
                missing_count=missing_count
            )
        else:
            return self._feedback_config.low_score
    
    def get_detailed_analysis(
        self, 
        question_id: int, 
        user_answer: str
    ) -> Dict:
        """
        Get detailed NLI analysis for debugging or explanation
        
        Args:
            question_id: Question ID
            user_answer: Student answer
            
        Returns:
            Detailed analysis with per-key-point scores
        """
        question = self._question_service.get_question_by_id(question_id)
        if not question:
            raise HTTPException(status_code=404, detail="Question not found")
        
        key_points = question["key_points"]
        sentences = self._text_processor.split_into_sentences(user_answer)
        
        detailed_results = []
        
        for key_point in key_points:
            nli_result = self._nli_service.evaluate_answer_against_keypoint(
                student_answer=user_answer,
                key_point=key_point["text"],
                sentences=sentences
            )
            
            detailed_results.append({
                "key_point": key_point["text"],
                "is_covered": nli_result["is_covered"],
                "confidence": round(nli_result["confidence"], 3),
                "best_entailment": round(nli_result["best_entailment"], 3),
                "has_contradiction": nli_result["has_contradiction"],
                "sentence_scores": [
                    {
                        "sentence": detail["sentence"],
                        "entailment": round(detail["entailment"], 3),
                        "contradiction": round(detail["contradiction"], 3)
                    }
                    for detail in nli_result["details"]
                ]
            })
        
        return {
            "question_id": question_id,
            "question_text": question["question_text"],
            "user_answer": user_answer,
            "key_points_analysis": detailed_results,
            "summary": {
                "total_key_points": len(key_points),
                "covered_count": sum(1 for r in detailed_results if r["is_covered"]),
                "contradiction_count": sum(1 for r in detailed_results if r["has_contradiction"])
            }
        }
