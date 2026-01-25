"""
Pydantic models for request/response schemas with comprehensive Swagger documentation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QuestionResponse(BaseModel):
    """Response model for a single question"""
    question_id: int = Field(..., description="Unique identifier for the question", example=1)
    question_text: str = Field(..., description="The question text to be answered", example="What is inflation?")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question_id": 1,
                "question_text": "What is inflation?"
            }
        }


class AnswerRequest(BaseModel):
    """Request model for grading an answer"""
    question_id: int = Field(..., description="The ID of the question being answered", example=1, ge=1)
    user_answer: str = Field(
        ..., 
        description="The student's answer text",
        example="Inflation is a general increase in prices over time, reducing the purchasing power of money",
        min_length=10
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question_id": 1,
                "user_answer": "Inflation is a general increase in prices over time, reducing the purchasing power of money"
            }
        }


class AnswerResponse(BaseModel):
    """Response model for a graded answer"""
    score: float = Field(..., description="Score as percentage (0-100)", ge=0, le=100, example=75.0)
    hit_key_points: List[str] = Field(
        ..., 
        description="Key points that the student's answer covers",
        example=["General increase in prices", "Reduction of purchasing power"]
    )
    missing_key_points: List[str] = Field(
        ..., 
        description="Key points that the student's answer missed",
        example=[]
    )
    feedback: str = Field(
        ..., 
        description="Feedback message for the student",
        example="Correct! You covered all the key points."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 100.0,
                "hit_key_points": ["General increase in prices", "Reduction of purchasing power"],
                "missing_key_points": [],
                "feedback": "Correct! You covered all the key points."
            }
        }


class KeyPoint(BaseModel):
    """Model representing a key point in a question's rubric"""
    text: str = Field(..., description="The key point text", example="General increase in prices")
    weight: int = Field(default=1, description="Weight of this key point in scoring", ge=1, example=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "General increase in prices",
                "weight": 1
            }
        }


class Question(BaseModel):
    """Complete question model with key points"""
    question_id: int = Field(..., description="Unique identifier for the question", example=1)
    question_text: str = Field(..., description="The question text", example="What is inflation?")
    key_points: List[KeyPoint] = Field(
        ..., 
        description="List of key points that should be in a correct answer",
        example=[
            {"text": "General increase in prices", "weight": 1},
            {"text": "Reduction of purchasing power", "weight": 1}
        ]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "question_id": 1,
                "question_text": "What is inflation?",
                "key_points": [
                    {"text": "General increase in prices", "weight": 1},
                    {"text": "Reduction of purchasing power", "weight": 1}
                ]
            }
        }


class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    message: str = Field(..., description="Health status message", example="Answer Evaluator API is running")
    questions_loaded: int = Field(..., description="Number of questions loaded in the system", example=42, ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Answer Evaluator API is running",
                "questions_loaded": 42
            }
        }


class AllQuestionsResponse(BaseModel):
    """Response model for retrieving all questions"""
    questions: List[Question] = Field(
        ..., 
        description="List of all available questions with their key points"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "questions": [
                    {
                        "question_id": 1,
                        "question_text": "What is inflation?",
                        "key_points": [
                            {"text": "General increase in prices", "weight": 1},
                            {"text": "Reduction of purchasing power", "weight": 1}
                        ]
                    }
                ]
            }
        }


class CacheStatsResponse(BaseModel):
    """Response model for cache statistics"""
    question_id: Optional[int] = Field(None, description="Question ID if filtered", example=1)
    cached_answers_count: int = Field(..., description="Number of cached answers", example=47, ge=0)
    average_score: float = Field(..., description="Average score of cached answers", example=72.5, ge=0, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question_id": 1,
                "cached_answers_count": 47,
                "average_score": 72.5
            }
        }


class CacheClearResponse(BaseModel):
    """Response model for cache clearing operation"""
    message: str = Field(..., description="Success message", example="Cache cleared for question 1")
    question_id: int = Field(..., description="The question ID that was cleared", example=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Cache cleared for question 1",
                "question_id": 1
            }
        }


class ErrorResponse(BaseModel):
    """Response model for error cases"""
    detail: str = Field(..., description="Error message", example="Question not found")
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Question not found"
            }
        }
