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
    AllQuestionsResponse,
    AddQuestionRequest,
    AddQuestionResponse,
    DeleteQuestionResponse
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
async def grade_answer_endpoint(
    request: AnswerRequest,
    grading_service: GradingService = Depends(get_grading_service)
) -> AnswerResponse:
    """
    Grade a user's answer against the question's key points
    
    This endpoint:
    1. Validates the answer format and content
    2. Gets embedding for user answer
    3. Compares with precomputed key point embeddings
    4. Returns grade with score and feedback
    
    Args:
        request: AnswerRequest with question_id and user_answer
        
    Returns:
        AnswerResponse with score, hit/missing points, and feedback
        
    Raises:
        HTTPException: If question not found or grading fails
    """
    return grading_service.grade_answer(request.question_id, request.user_answer)


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


@router.post("/question", response_model=AddQuestionResponse)
async def add_question(
    request: AddQuestionRequest,
    q_service: QuestionService = Depends(get_question_service),
    g_service: GradingService = Depends(get_grading_service)
) -> AddQuestionResponse:
    """
    Add a new question with key points to the database and vector store
    
    This endpoint:
    1. Adds the question to the questions bank
    2. Computes embeddings for all key points
    3. Stores embeddings in Pinecone vector database
    
    Args:
        request: AddQuestionRequest with question_text and key_points
        
    Returns:
        AddQuestionResponse with question_id and confirmation message
        
    Raises:
        HTTPException: If question is invalid or embedding computation fails
    """
    try:
        # Validate request
        if not request.question_text or not request.question_text.strip():
            raise HTTPException(status_code=400, detail="Question text cannot be empty")
        
        if not request.key_points or len(request.key_points) == 0:
            raise HTTPException(status_code=400, detail="At least one key point is required")
        
        # Convert key points to dictionary format
        key_points_dict = [
            {"text": kp.text, "weight": kp.weight}
            for kp in request.key_points
        ]
        
        # Add question to questions bank
        question_id = q_service.add_question(
            question_text=request.question_text.strip(),
            key_points=key_points_dict
        )
        
        # Get the newly added question
        new_question = q_service.get_question_by_id(question_id)
        
        if not new_question:
            raise HTTPException(status_code=500, detail="Failed to retrieve newly added question")
        
        # Compute and store embeddings
        print(f"🔄 Computing embeddings for new question {question_id}...")
        success = g_service.compute_and_store_embeddings_for_question(new_question)
        
        if not success:
            print(f"⚠️ Warning: Failed to store embeddings in Pinecone for question {question_id}")
            # Question is still added to the bank, embeddings will be computed on-demand
        
        return AddQuestionResponse(
            question_id=question_id,
            question_text=request.question_text.strip(),
            message=f"Question added successfully with {len(key_points_dict)} key point(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question: {str(e)}")


@router.delete("/question/{question_id}", response_model=DeleteQuestionResponse)
async def delete_question(
    question_id: int,
    q_service: QuestionService = Depends(get_question_service),
    g_service: GradingService = Depends(get_grading_service)
) -> DeleteQuestionResponse:
    """
    Delete a question from the database and remove its embeddings from the vector store
    
    This endpoint:
    1. Removes the question from the questions bank
    2. Removes embeddings from memory
    3. Deletes embeddings from Pinecone vector database
    
    Args:
        question_id: ID of the question to delete
        
    Returns:
        DeleteQuestionResponse with confirmation message
        
    Raises:
        HTTPException: If question not found or deletion fails
    """
    try:
        # Check if question exists
        if not q_service.question_exists(question_id):
            raise HTTPException(status_code=404, detail=f"Question with ID {question_id} not found")
        
        # Remove question from questions bank
        removed = q_service.remove_question(question_id)
        
        if not removed:
            raise HTTPException(status_code=500, detail=f"Failed to remove question {question_id} from questions bank")
        
        # Remove embeddings from memory and Pinecone
        print(f"🔄 Removing embeddings for question {question_id}...")
        embedding_removed = g_service.remove_question_embeddings(question_id)
        
        if not embedding_removed:
            print(f"⚠️ Warning: Failed to remove embeddings from Pinecone for question {question_id}")
            # Question is still removed from bank, but embeddings might remain in Pinecone
        
        return DeleteQuestionResponse(
            question_id=question_id,
            message=f"Question {question_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete question: {str(e)}")
