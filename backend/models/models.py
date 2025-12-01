"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel
from typing import List, Dict, Any


class QuestionResponse(BaseModel):
    question_id: int
    question_text: str


class AnswerRequest(BaseModel):
    question_id: int
    user_answer: str


class AnswerResponse(BaseModel):
    score: float
    hit_key_points: List[str]
    missing_key_points: List[str]
    feedback: str


class KeyPoint(BaseModel):
    text: str
    weight: int = 1


class Question(BaseModel):
    question_id: int
    question_text: str
    key_points: List[KeyPoint]
    category: str = "unknown"  # Category/source of the question (e.g., "biology", "economics", "user")


class HealthCheckResponse(BaseModel):
    message: str
    questions_loaded: int


class AllQuestionsResponse(BaseModel):
    questions: List[Question]


class KeyPointInput(BaseModel):
    text: str
    weight: int = 1


class AddQuestionRequest(BaseModel):
    question_text: str
    key_points: List[KeyPointInput]


class AddQuestionResponse(BaseModel):
    question_id: int
    question_text: str
    message: str


class DeleteQuestionResponse(BaseModel):
    message: str
    question_id: int
