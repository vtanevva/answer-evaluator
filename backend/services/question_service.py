import os
import json
import random
from typing import List, Dict, Optional
from pathlib import Path

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
        self._user_added_question_ids: set = set()  # Track IDs of user-added questions
    
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
            fallback_questions = self._get_fallback_questions()
            # Add category to fallback questions
            for question in fallback_questions:
                question['category'] = 'fallback'
            self._questions_bank = fallback_questions
            print(f"⚠️ Using fallback questions. Loaded {len(self._questions_bank)} questions")
        
        # Create lookup dictionary for faster access
        self._questions_by_id = {
            question["question_id"]: question 
            for question in self._questions_bank
        }
        
        self._is_loaded = True
    
    def _resolve_questions_directory_path(self, directory_path: str) -> str:
        """
        Resolve the questions directory path to an absolute path
        
        Args:
            directory_path: Relative or absolute path to questions directory
            Example: "../data/questions/" should resolve to project_root/data/questions/
            
        Returns:
            Resolved absolute path to questions directory at project root
        """
        # Get the backend directory path (answer-evaluator/backend/)
        backend_dir = Path(__file__).parent.parent
        
        # Get project root (answer-evaluator/) by going up one level from backend
        project_root = backend_dir.parent
        
        # If path is relative (starts with ../), it means go up from backend to project root
        if directory_path.startswith("../"):
            # Remove the "../" prefix (which means "go up from backend to project root")
            relative_path = directory_path[3:]  # Remove "../" (3 characters)
            resolved_path = project_root / relative_path  # project_root/data/questions/
        elif not os.path.isabs(directory_path):
            # Relative path - assume it's from project root
            resolved_path = project_root / directory_path
        else:
            # Already absolute path
            resolved_path = Path(directory_path)
        
        return str(resolved_path)
    
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
            # Resolve path to absolute path
            resolved_path = self._resolve_questions_directory_path(directory_path)
            print(f"📂 Resolved questions directory path: {resolved_path}")
            
            if not os.path.isdir(resolved_path):
                print(f"❌ Directory {resolved_path} not found")
                return all_questions
            
            # List all JSON files in the directory
            question_files = [
                file_name for file_name in os.listdir(resolved_path)
                if file_name.endswith('.json')
            ]
            
            if not question_files:
                print(f"⚠️ No JSON files found in {resolved_path}")
                return all_questions
            
            print(f"📁 Found {len(question_files)} JSON file(s) in {resolved_path}")
            
            # Load questions from each file (except user_added.json, which we'll load separately)
            for file_name in question_files:
                if file_name != 'user_added.json':  # Skip user_added.json here
                    file_path = os.path.join(resolved_path, file_name)
                    questions = load_questions_from_file(file_path)
                    
                    # Extract category name from filename (e.g., "economics.json" -> "economics")
                    category = os.path.splitext(file_name)[0]
                    
                    # Add category to each question
                    for question in questions:
                        question['category'] = category
                    
                    all_questions.extend(questions)
            
            # Load user-added questions separately (at the end to preserve IDs)
            user_added_file = os.path.join(resolved_path, 'user_added.json')
            if os.path.exists(user_added_file):
                user_questions = load_questions_from_file(user_added_file)
                if user_questions:
                    # Add category for user-added questions
                    for q in user_questions:
                        q['category'] = 'user'
                        self._user_added_question_ids.add(q.get('question_id'))
                    all_questions.extend(user_questions)
                    print(f"📝 Loaded {len(user_questions)} user-added question(s) from user_added.json")
            
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
    
    def add_question(self, question_text: str, key_points: List[Dict]) -> int:
        """
        Add a new question to the questions bank with auto-generated ID
        
        Args:
            question_text: The question text
            key_points: List of key point dictionaries with 'text' and 'weight' keys
            
        Returns:
            The question_id of the newly added question
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        # Generate next available question ID
        if self._questions_bank:
            max_id = max(question["question_id"] for question in self._questions_bank)
            new_question_id = max_id + 1
        else:
            new_question_id = 1
        
        # Create new question dictionary
        new_question = {
            "question_id": new_question_id,
            "question_text": question_text,
            "key_points": key_points,
            "category": "user"  # Mark as user-added category
        }
        
        # Add to questions bank
        self._questions_bank.append(new_question)
        self._questions_by_id[new_question_id] = new_question
        self._user_added_question_ids.add(new_question_id)  # Mark as user-added
        
        print(f"✅ Added new question with ID {new_question_id}: {question_text[:50]}...")
        
        # Save to file for persistence
        self._save_user_added_questions()
        
        return new_question_id
    
    def remove_question(self, question_id: int) -> bool:
        """
        Remove a question from the questions bank
        
        Args:
            question_id: ID of the question to remove
            
        Returns:
            True if question was removed, False if question doesn't exist
        """
        if not self._is_loaded:
            self.load_questions_bank()
        
        if question_id not in self._questions_by_id:
            print(f"⚠️ Question {question_id} not found in questions bank")
            return False
        
        # Remove from questions bank
        self._questions_bank = [
            q for q in self._questions_bank 
            if q["question_id"] != question_id
        ]
        
        # Remove from lookup dictionary
        del self._questions_by_id[question_id]
        
        # Remove from user-added tracking if it was user-added
        self._user_added_question_ids.discard(question_id)
        
        print(f"✅ Removed question {question_id} from questions bank")
        
        # Update file for persistence
        self._save_user_added_questions()
        
        return True
    
    def _get_user_added_questions_file(self) -> str:
        """
        Get the path to the user-added questions file
        
        Returns:
            Path to user_added.json file in data/questions/ directory at project root
        """
        directory_path = settings.questions.default_file_path
        
        # Resolve path using the same method as loading questions
        resolved_path = self._resolve_questions_directory_path(directory_path)
        print(f"💾 User-added questions will be saved to: {resolved_path}/user_added.json")
        
        # Ensure directory exists
        os.makedirs(resolved_path, exist_ok=True)
        
        # Return path to user_added.json
        return os.path.join(resolved_path, "user_added.json")
    
    def _save_user_added_questions(self) -> None:
        """
        Save user-added questions to user_added.json file for persistence
        
        This separates user-added questions from pre-loaded questions in JSON files.
        """
        try:
            user_added_file = self._get_user_added_questions_file()
            
            # Find user-added questions using tracked IDs
            user_added_questions = [
                question for question in self._questions_bank
                if question["question_id"] in self._user_added_question_ids
            ]
            
            # Save to file
            with open(user_added_file, 'w', encoding='utf-8') as f:
                json.dump(user_added_questions, f, indent=2, ensure_ascii=False)
            
            if user_added_questions:
                print(f"💾 Saved {len(user_added_questions)} user-added question(s) to {user_added_file}")
            else:
                # If no user-added questions, remove the file
                if os.path.exists(user_added_file):
                    os.remove(user_added_file)
                    print(f"🗑️ Removed empty user_added.json file")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to save user-added questions to file: {e}")
            # Don't raise - allow question to be added even if file save fails