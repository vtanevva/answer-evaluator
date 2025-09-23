import os
import json
import random
from typing import List, Dict, Optional

from models.models import Question, KeyPoint
from core.config import settings
from load_questions import load_questions_from_file


class QuestionService:
    """
    Service for managing question data and operations
    
    This service handles:
    - Loading questions from file or fallback data
    - Managing question lookup by ID
    - Providing random question selection
    """
    
    def __init__(self):
        """Initialize question service"""
        self._questions_bank: List[Dict] = []
        self._questions_by_id: Dict[int, Dict] = {}
        self._is_loaded = False
    
    def load_questions_bank(self) -> None:
        """
        Load questions from file or use fallback data
        
        Tries to load from the configured file path first,
        then falls back to hardcoded questions if file loading fails.
        """
        file_path = settings.questions.default_file_path
        loaded_questions = load_questions_from_file(file_path)
        
        if loaded_questions:
            self._questions_bank = loaded_questions
            print(f"✅ Loaded {len(self._questions_bank)} questions from {file_path}")
        else:
            # Use fallback questions from configuration
            self._questions_bank = self._get_fallback_questions()
            print(f"⚠️ Using fallback questions. Loaded {len(self._questions_bank)} questions")
        
        # Create lookup dictionary for faster access
        self._questions_by_id = {
            question["question_id"]: question 
            for question in self._questions_bank
        }
        
        self._is_loaded = True
    
    def _get_fallback_questions(self) -> List[Dict]:
        """
        Get fallback questions from configuration
        
        Returns:
            List of fallback question dictionaries
        """
        return settings.questions.fallback_questions
    
    def get_all_questions(self) -> List[Dict]:
        """
        Get all loaded questions
        
        Returns:
            List of all question dictionaries
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        return self._questions_bank
    
    def get_question_by_id(self, question_id: int) -> Optional[Dict]:
        """
        Get a question by its ID
        
        Args:
            question_id: ID of the question to retrieve
            
        Returns:
            Question dictionary if found, None otherwise
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        return self._questions_by_id.get(question_id)
    
    def get_random_question(self) -> Dict:
        """
        Get a random question from the loaded questions
        
        Returns:
            Random question dictionary
            
        Raises:
            ValueError: If no questions are loaded
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        if not self._questions_bank:
            raise ValueError("No questions loaded")
        
        return random.choice(self._questions_bank)
    
    def get_questions_count(self) -> int:
        """
        Get the total number of loaded questions
        
        Returns:
            Number of questions loaded
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        return len(self._questions_bank)
    
    def question_exists(self, question_id: int) -> bool:
        """
        Check if a question with the given ID exists
        
        Args:
            question_id: ID to check
            
        Returns:
            True if question exists, False otherwise
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        return question_id in self._questions_by_id
