"""
API routes for the Answer Evaluator application
"""

import random
from fastapi import APIRouter, HTTPException, Depends, status

from models.models import (
    QuestionResponse, 
    AnswerRequest, 
    AnswerResponse, 
    HealthCheckResponse,
    AllQuestionsResponse,
    CacheStatsResponse,
    CacheClearResponse,
    ErrorResponse
)
from services.question_service import QuestionService
from services.grading_service import GradingService


# Create router instance
router = APIRouter()

# Global service instances (will be initialized in main.py)
question_service: QuestionService = None
grading_service: GradingService = None


def get_question_service() -> QuestionService:
    """Dependency to get question service instance"""
    if question_service is None:
        raise HTTPException(status_code=500, detail="Question service not initialized")
    return question_service


def get_grading_service() -> GradingService:
    """Dependency to get grading service instance"""
    if grading_service is None:
        raise HTTPException(status_code=500, detail="Grading service not initialized")
    return grading_service


# ============================================================================
# HEALTH & STATUS ENDPOINTS
# ============================================================================

@router.get(
    "/",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the API is running and see how many questions are loaded",
    tags=["Health"]
)
async def health_check(
    q_service: QuestionService = Depends(get_question_service)
) -> HealthCheckResponse:
    """
    Health check endpoint to verify API is operational
    
    Returns:
        HealthCheckResponse with status message and questions count
    """
    return HealthCheckResponse(
        message="Answer Evaluator API is running",
        questions_loaded=q_service.get_questions_count()
    )


# ============================================================================
# QUESTION ENDPOINTS
# ============================================================================

@router.get(
    "/question",
    response_model=QuestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get random question",
    description="Retrieve a random question from the question bank",
    tags=["Questions"],
    responses={
        200: {
            "description": "A random question",
            "model": QuestionResponse
        },
        500: {
            "description": "No questions loaded",
            "model": ErrorResponse
        }
    }
)
async def get_random_question(
    q_service: QuestionService = Depends(get_question_service)
) -> QuestionResponse:
    """
    Get a random question from the question bank
    
    This endpoint returns a random question to present to the student.
    Useful for quiz-like interfaces.
    
    Returns:
        QuestionResponse with question_id and question_text
        
    Raises:
        HTTPException: If no questions are loaded in the system
    """
    try:
        question = q_service.get_random_question()
        
        return QuestionResponse(
            question_id=question["question_id"],
            question_text=question["question_text"]
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/questions",
    response_model=AllQuestionsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get all questions",
    description="Retrieve all available questions with their rubrics (key points)",
    tags=["Questions"],
    responses={
        200: {
            "description": "List of all questions with key points",
            "model": AllQuestionsResponse
        }
    }
)
async def get_all_questions(
    q_service: QuestionService = Depends(get_question_service)
) -> AllQuestionsResponse:
    """
    Get all questions from the question bank
    
    This endpoint returns all available questions along with their key points
    (rubric). Useful for debugging and development, or for displaying all
    available questions to students.
    
    Returns:
        AllQuestionsResponse with complete list of questions and their key points
    """
    questions = q_service.get_all_questions()
    
    return AllQuestionsResponse(questions=questions)


# ============================================================================
# GRADING ENDPOINTS
# ============================================================================

@router.post(
    "/answer",
    response_model=AnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Grade student answer",
    description="Grade a student's answer against the question's key points using embeddings and semantic analysis",
    tags=["Grading"],
    responses={
        200: {
            "description": "Successfully graded answer",
            "model": AnswerResponse
        },
        404: {
            "description": "Question not found",
            "model": ErrorResponse
        },
        400: {
            "description": "Invalid answer (too short or invalid content)",
            "model": ErrorResponse
        },
        500: {
            "description": "Grading failed",
            "model": ErrorResponse
        }
    }
)
async def grade_answer_endpoint(
    request: AnswerRequest,
    grading_service: GradingService = Depends(get_grading_service)
) -> AnswerResponse:
    """
    Grade a student's answer using semantic similarity
    
    This endpoint implements the complete grading pipeline:
    
    1. **Fast Path (Answer Cache)**: If an identical/very similar answer was graded before
       (similarity ≥ 0.99), returns cached result instantly (~10-50ms)
    
    2. **Normal Path (Full Evaluation)**:
       - Validates answer (minimum length, no invalid content)
       - Generates embeddings for student answer (sentence-level)
       - Compares against precomputed key point embeddings
       - Uses NLI (Natural Language Inference) for semantic verification (if enabled)
       - Calculates score based on key points covered
       - Caches result for future use
    
    **Grading Methods:**
    - **Embedding Mode**: Pure cosine similarity matching
    - **NLI Mode**: Semantic inference verification
    - **Hybrid Mode** (default): Combines cosine similarity with NLI for high accuracy
    
    **Score Calculation:**
    - 0-100 percentage based on fraction of key points covered
    - Considers semantic similarity, lexical overlap, and contradiction detection
    
    Args:
        request: AnswerRequest containing question_id and user_answer
        grading_service: Grading service instance (injected)
        
    Returns:
        AnswerResponse with:
        - score: Percentage (0-100)
        - hit_key_points: Key points covered by the answer
        - missing_key_points: Key points not covered
        - feedback: Constructive feedback for the student
        
    Raises:
        HTTPException (404): Question not found
        HTTPException (400): Answer validation failed (too short/invalid)
        HTTPException (500): Grading computation failed
    """
    return grading_service.grade_answer(request.question_id, request.user_answer)


# ============================================================================
# CACHE MANAGEMENT ENDPOINTS
# ============================================================================

@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get cache statistics",
    description="Retrieve statistics about cached student answers",
    tags=["Cache Management"],
    responses={
        200: {
            "description": "Cache statistics",
            "model": CacheStatsResponse
        },
        500: {
            "description": "Failed to retrieve cache stats",
            "model": ErrorResponse
        }
    }
)
async def get_cache_stats(
    question_id: int = None,
    grading_service: GradingService = Depends(get_grading_service)
) -> CacheStatsResponse:
    """
    Get statistics about the answer cache
    
    Retrieve information about how many answers have been cached and their
    average scores. Useful for monitoring cache effectiveness and understanding
    common student answer patterns.
    
    **Answer Caching:**
    When a student submits an answer, it's embedded and stored in Pinecone.
    If another student submits a very similar answer (≥0.99 similarity),
    the cached grade is returned instantly instead of recomputing.
    
    Benefits:
    - 10-200x faster grading for repeated answers
    - Significantly reduced API costs
    - Insights into common student answers
    
    Args:
        question_id: Optional question ID to filter statistics. If not provided,
                    returns overall cache statistics
        grading_service: Grading service instance (injected)
        
    Returns:
        CacheStatsResponse with:
        - question_id: The question ID (if filtered)
        - cached_answers_count: Number of unique cached answers
        - average_score: Average score of cached answers
        
    Raises:
        HTTPException (500): Failed to retrieve cache statistics
    """
    stats = grading_service.get_answer_cache_stats(question_id)
    if stats is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve cache stats"
        )
    return CacheStatsResponse(**stats)


@router.delete(
    "/cache/question/{question_id}",
    response_model=CacheClearResponse,
    status_code=status.HTTP_200_OK,
    summary="Clear cache for question",
    description="Clear all cached answers for a specific question",
    tags=["Cache Management"],
    responses={
        200: {
            "description": "Cache cleared successfully",
            "model": CacheClearResponse
        },
        500: {
            "description": "Failed to clear cache",
            "model": ErrorResponse
        }
    }
)
async def clear_cache_for_question(
    question_id: int,
    grading_service: GradingService = Depends(get_grading_service)
) -> CacheClearResponse:
    """
    Clear cached answers for a specific question
    
    Use this endpoint when:
    - Question content is updated or corrected
    - Key points are modified
    - You need to ensure fresh evaluation for all similar answers
    - Debugging grading issues related to specific questions
    
    **Important:** Clearing the cache means the next students answering this
    question will require full evaluation computation until new answers are
    cached again.
    
    Args:
        question_id: ID of the question to clear cache for
        grading_service: Grading service instance (injected)
        
    Returns:
        CacheClearResponse with success message
        
    Raises:
        HTTPException (500): Failed to clear cache
    """
    success = grading_service.clear_answer_cache_for_question(question_id)
    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache for question {question_id}"
        )
    
    return CacheClearResponse(
        message=f"Cache cleared for question {question_id}",
        question_id=question_id
    )
