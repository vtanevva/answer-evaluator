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
        Load all questions from directory or use fallback data
        
        Lists the default_file_path directory and loads all question files found in it.
        Falls back to hardcoded questions if no files are found or an error occurs.
        """
        directory_path = settings.questions.default_file_path
        loaded_questions = self._load_all_questions_from_directory(directory_path)
        
        if loaded_questions:
            self._questions_bank = loaded_questions
            print(f"✅ Loaded {len(self._questions_bank)} questions from {directory_path}")
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
    
    def _load_all_questions_from_directory(self, directory_path: str) -> List[Dict]:
        """
        Load all question files from a directory
        
        Lists all JSON files in the directory and loads questions from each file.
        
        Args:
            directory_path: Path to the directory containing question files
            
        Returns:
            List of all questions loaded from all files, or empty list if directory
            doesn't exist or no files are found
        """
        all_questions = []
        
        try:
            if not os.path.isdir(directory_path):
                print(f"❌ Directory {directory_path} not found")
                return all_questions
            
            # List all JSON files in the directory
            question_files = [
                file_name for file_name in os.listdir(directory_path)
                if file_name.endswith('.json')
            ]
            
            if not question_files:
                print(f"⚠️ No JSON files found in {directory_path}")
                return all_questions
            
            print(f"📁 Found {len(question_files)} JSON file(s) in {directory_path}")
            
            # Load questions from each file
            for file_name in question_files:
                file_path = os.path.join(directory_path, file_name)
                questions = load_questions_from_file(file_path)
                all_questions.extend(questions)
            
            return all_questions
            
        except Exception as e:
            print(f"❌ Error loading questions from directory {directory_path}: {e}")
            return all_questions
    
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
