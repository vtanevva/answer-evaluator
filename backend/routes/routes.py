"""
API routes for the Answer Evaluator application
"""

import random
from fastapi import APIRouter, HTTPException, Depends

from models.models import (
    QuestionResponse, 
    AnswerRequest, 
    AnswerResponse, 
    HealthCheckResponse,
    AllQuestionsResponse
)
from services.question_service import QuestionService
from services.evaluation_service import EvaluationService


# Create router instance
router = APIRouter()

# Global service instances (will be initialized in main.py)
question_service: QuestionService = None
evaluation_service: EvaluationService = None


def get_question_service() -> QuestionService:
    """Dependency to get question service instance"""
    if question_service is None:
        raise HTTPException(status_code=500, detail="Question service not initialized")
    return question_service


def get_evaluation_service() -> EvaluationService:
    """Dependency to get evaluation service instance"""
    if evaluation_service is None:
        raise HTTPException(status_code=500, detail="Evaluation service not initialized")
    return evaluation_service


@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    q_service: QuestionService = Depends(get_question_service)
) -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        Health status and number of questions loaded
    """
    return HealthCheckResponse(
        message="Answer Evaluator API is running",
        questions_loaded=q_service.get_questions_count()
    )


@router.get("/question", response_model=QuestionResponse)
async def get_random_question(
    q_service: QuestionService = Depends(get_question_service)
) -> QuestionResponse:
    """
    Get a random question from the question bank
    
    Returns:
        QuestionResponse with question_id and question_text
        
    Raises:
        HTTPException: If no questions are loaded
    """
    try:
        question = q_service.get_random_question()
        
        return QuestionResponse(
            question_id=question["question_id"],
            question_text=question["question_text"]
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=AnswerResponse)
async def evaluate_answer_endpoint(
    request: AnswerRequest,
    eval_service: EvaluationService = Depends(get_evaluation_service)
) -> AnswerResponse:
    """
    Evaluate a user's answer against the question's key points
    
    This endpoint:
    1. Validates the answer format and content
    2. Gets embedding for user answer
    3. Compares with precomputed key point embeddings
    4. Returns evaluation results with score and feedback
    
    Args:
        request: AnswerRequest with question_id and user_answer
        
    Returns:
        AnswerResponse with score, hit/missing points, and feedback
        
    Raises:
        HTTPException: If question not found or evaluation fails
    """
    return eval_service.evaluate_answer(request.question_id, request.user_answer)


@router.get("/questions", response_model=AllQuestionsResponse)
async def get_all_questions(
    q_service: QuestionService = Depends(get_question_service)
) -> AllQuestionsResponse:
    """
    Get all questions (for debugging/development)
    
    Returns:
        AllQuestionsResponse with list of all questions
    """
    questions = q_service.get_all_questions()
    
    return AllQuestionsResponse(questions=questions)
