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


class HealthCheckResponse(BaseModel):
    message: str
    questions_loaded: int


class AllQuestionsResponse(BaseModel):
    questions: List[Question]
