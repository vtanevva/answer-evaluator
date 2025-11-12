"""
Unified mistake analysis using the SAME services as evaluation_script.py
This ensures we see exactly what the evaluation script sees
"""

import os
import sys
from dotenv import load_dotenv
import json
from pathlib import Path

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.grading_service import GradingService
from services.question_service import QuestionService
import openai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

class UnifiedMistakeAnalysis:
    """Use the EXACT same logic as EvaluationBenchmark"""
    
    def __init__(self):
        """Initialize using the same logic as evaluation_script.py"""
        self._load_questions_from_data_files()
        if len(self._questions) == 0:
            print("❌ No questions loaded")
            return
        self._initialize_grading_service()
        
    def _load_questions_from_data_files(self) -> None:
        """Load questions from annotated questions file - EXACT same as evaluation_script.py"""
        # Load from our annotated questions file with realistic student answers
        annotated_file = Path(__file__).parent / "annotated_questions.json"
        self._questions = []
        
        if annotated_file.exists():
            with open(annotated_file, 'r', encoding='utf-8') as f:
                self._questions = json.load(f)
                print(f"✅ Loaded {len(self._questions)} annotated questions with realistic student answers")
        else:
            print(f"❌ Annotated questions file not found: {annotated_file}")
    
    def _initialize_grading_service(self) -> None:
        """Initialize grading service - EXACT same as evaluation_script.py"""
        # Check for OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        print(f"✅ OpenAI API key found")
        openai.api_key = openai_api_key
        
        # Create question service with our loaded questions - EXACT same as evaluation_script.py
        self._question_service = QuestionService()
        # Override with our annotated questions data
        self._question_service._questions_bank = self._questions  # Use _questions_bank instead of _questions
        self._question_service._questions_by_id = {q["question_id"]: q for q in self._questions}
        
        # CRITICAL: Mark as loaded to prevent reloading fallback questions
        self._question_service._is_loaded = True
        
        # Ensure the question service uses our data
        print(f"✅ Loaded {len(self._questions)} questions for evaluation")
        for q in self._questions[:3]:  # Show first 3
            print(f"  Question {q['question_id']}: {q['question_text'][:50]}...")
            print(f"    Key Points: {[kp['text'][:30] + '...' for kp in q['key_points']]}")
        
        # Create evaluation service - EXACT same as evaluation_script.py
        self._grading_service = GradingService(
            self._question_service, 
            openai.OpenAI(api_key=openai_api_key)
        )
        self._grading_service.precompute_embeddings()
        
        print(f"✅ Grading service initialized with correct questions")

    def analyze_detailed_mistakes(self):
        """Analyze each answer in detail to show exact mistakes"""
        
        print("🔍 UNIFIED DETAILED MISTAKE ANALYSIS")
        print("=" * 80)
        
        # Analyze each question in detail
        for question in self._questions[:3]:  # First 3 questions only for detailed analysis
            self.analyze_question_detailed(question)

    def analyze_question_detailed(self, question):
        """Analyze a single question in detail"""
        
        question_id = question["question_id"]
        question_text = question["question_text"]
        
        print(f"\n📝 QUESTION {question_id}: {question_text}")
        print("=" * 80)
        
        print(f"Key Points:")
        for i, kp in enumerate(question["key_points"]):
            print(f"  {i+1}. {kp['text']}")
        
        print(f"\n🎯 CLASSIFICATION THRESHOLD: 50% (answers >= 50% score are considered correct)")
        
        # Analyze correct answers
        print(f"\n✅ ANALYZING CORRECT ANSWERS:")
        print("-" * 60)
        
        correct_correctly_classified = 0
        correct_incorrectly_classified = 0
        
        for i, answer in enumerate(question["correct_answers"]):
            print(f"\n  Correct Answer {i+1}:")
            print(f"  Text: '{answer}'")
            
            try:
                result = self._grading_service.grade_answer(question_id, answer)
                score = result.score
                is_correct = score >= 50.0  # Classification threshold
                
                status = "✅ CORRECTLY CLASSIFIED" if is_correct else "❌ INCORRECTLY CLASSIFIED"
                print(f"  Score: {score}%")
                print(f"  Classification: {status}")
                print(f"  Hit Key Points: {result.hit_key_points}")
                print(f"  Missing Key Points: {result.missing_key_points}")
                
                if is_correct:
                    correct_correctly_classified += 1
                else:
                    correct_incorrectly_classified += 1
                    print(f"  🚨 MISTAKE: Correct answer marked as incorrect!")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                correct_incorrectly_classified += 1
        
        # Analyze incorrect answers
        print(f"\n❌ ANALYZING INCORRECT ANSWERS:")
        print("-" * 60)
        
        incorrect_correctly_classified = 0
        incorrect_incorrectly_classified = 0
        
        for i, answer in enumerate(question["incorrect_answers"]):
            print(f"\n  Incorrect Answer {i+1}:")
            print(f"  Text: '{answer}'")
            
            try:
                result = self._grading_service.grade_answer(question_id, answer)
                score = result.score
                is_correct = score >= 50.0  # Classification threshold
                
                status = "✅ CORRECTLY CLASSIFIED" if not is_correct else "❌ INCORRECTLY CLASSIFIED"
                print(f"  Score: {score}%")
                print(f"  Classification: {status}")
                print(f"  Hit Key Points: {result.hit_key_points}")
                print(f"  Missing Key Points: {result.missing_key_points}")
                
                if not is_correct:
                    incorrect_correctly_classified += 1
                else:
                    incorrect_incorrectly_classified += 1
                    print(f"  🚨 MISTAKE: Incorrect answer marked as correct!")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                incorrect_incorrectly_classified += 1
        
        # Summary for this question
        total_correct_answers = len(question["correct_answers"])
        total_incorrect_answers = len(question["incorrect_answers"])
        total_answers = total_correct_answers + total_incorrect_answers
        
        total_correctly_classified = correct_correctly_classified + incorrect_correctly_classified
        accuracy = (total_correctly_classified / total_answers) * 100 if total_answers > 0 else 0
        
        print(f"\n📊 QUESTION {question_id} SUMMARY:")
        print(f"  Correct Answers: {correct_correctly_classified}/{total_correct_answers} correctly classified")
        print(f"  Incorrect Answers: {incorrect_correctly_classified}/{total_incorrect_answers} correctly classified")
        print(f"  Overall Accuracy: {total_correctly_classified}/{total_answers} ({accuracy:.1f}%)")
        
        # Show specific mistakes
        if correct_incorrectly_classified > 0 or incorrect_incorrectly_classified > 0:
            print(f"\n🚨 MISTAKES FOUND:")
            if correct_incorrectly_classified > 0:
                print(f"  - {correct_incorrectly_classified} correct answers marked as incorrect (False Negatives)")
            if incorrect_incorrectly_classified > 0:
                print(f"  - {incorrect_incorrectly_classified} incorrect answers marked as correct (False Positives)")

if __name__ == "__main__":
    analysis = UnifiedMistakeAnalysis()
    if len(analysis._questions) > 0:
        analysis.analyze_detailed_mistakes()
    else:
        print("❌ No questions loaded, cannot proceed with analysis")
