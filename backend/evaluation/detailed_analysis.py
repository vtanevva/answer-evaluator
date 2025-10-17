"""
Detailed analysis of evaluation examples to understand mistakes
"""

import os
import sys
from dotenv import load_dotenv
import json

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.grading_service import GradingService
from services.question_service import QuestionService
from services.embedding_service import EmbeddingService
import openai

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))

def analyze_evaluation_examples():
    """Analyze specific examples to understand why the system makes mistakes"""
    
    print("🔍 DETAILED ANALYSIS OF EVALUATION EXAMPLES")
    print("=" * 80)
    
    # Initialize services
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    
    # Load the evaluation data
    with open('annotated_questions.json', 'r') as f:
        questions = json.load(f)
    
    # Create a mock question service with our data
    question_service = QuestionService()
    question_service._questions = questions
    question_service._questions_by_id = {q["question_id"]: q for q in questions}
    
    # Create grading service
    grading_service = GradingService(question_service, openai_client)
    grading_service.precompute_embeddings()
    
    print("✅ Services initialized")
    
    # Analyze a few specific examples
    analyze_specific_examples(grading_service, questions)

def analyze_specific_examples(grading_service, questions):
    """Analyze specific examples in detail"""
    
    print(f"\n📊 ANALYZING SPECIFIC EXAMPLES")
    print("=" * 80)
    
    # Example 1: Question 3 (Plants) - This had good results
    print(f"\n🌱 EXAMPLE 1: Question 3 - 'How do plants make their own food?'")
    print("-" * 60)
    
    question = questions[2]  # Question 3
    question_id = question["question_id"]
    
    print(f"Question: {question['question_text']}")
    print(f"Key Points:")
    for i, kp in enumerate(question["key_points"]):
        print(f"  {i+1}. {kp['text']}")
    
    # Test correct answers
    print(f"\n✅ TESTING CORRECT ANSWERS:")
    for i, answer in enumerate(question["correct_answers"][:2]):  # Just first 2
        print(f"\n  Correct Answer {i+1}:")
        print(f"  Text: '{answer}'")
        
        try:
            result = grading_service.grade_answer(question_id, answer)
            print(f"  Score: {result.score}%")
            print(f"  Hit Key Points: {result.hit_key_points}")
            print(f"  Missing Key Points: {result.missing_key_points}")
            
            # Show detailed similarity scores
            print(f"  Detailed Analysis:")
            for j, kp in enumerate(question["key_points"]):
                # This is a bit hacky but let's see the similarity
                print(f"    Key Point {j+1}: '{kp['text']}' - Need to check similarity")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test incorrect answers
    print(f"\n❌ TESTING INCORRECT ANSWERS:")
    for i, answer in enumerate(question["incorrect_answers"][:2]):  # Just first 2
        print(f"\n  Incorrect Answer {i+1}:")
        print(f"  Text: '{answer}'")
        
        try:
            result = grading_service.grade_answer(question_id, answer)
            print(f"  Score: {result.score}%")
            print(f"  Hit Key Points: {result.hit_key_points}")
            print(f"  Missing Key Points: {result.missing_key_points}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Example 2: Question 4 (Inflation) - This had issues
    print(f"\n💰 EXAMPLE 2: Question 4 - 'Explain inflation'")
    print("-" * 60)
    
    question = questions[3]  # Question 4
    question_id = question["question_id"]
    
    print(f"Question: {question['question_text']}")
    print(f"Key Points:")
    for i, kp in enumerate(question["key_points"]):
        print(f"  {i+1}. {kp['text']}")
    
    # Test correct answers
    print(f"\n✅ TESTING CORRECT ANSWERS:")
    for i, answer in enumerate(question["correct_answers"][:2]):
        print(f"\n  Correct Answer {i+1}:")
        print(f"  Text: '{answer}'")
        
        try:
            result = grading_service.grade_answer(question_id, answer)
            print(f"  Score: {result.score}%")
            print(f"  Hit Key Points: {result.hit_key_points}")
            print(f"  Missing Key Points: {result.missing_key_points}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Test incorrect answers
    print(f"\n❌ TESTING INCORRECT ANSWERS:")
    for i, answer in enumerate(question["incorrect_answers"][:2]):
        print(f"\n  Incorrect Answer {i+1}:")
        print(f"  Text: '{answer}'")
        
        try:
            result = grading_service.grade_answer(question_id, answer)
            print(f"  Score: {result.score}%")
            print(f"  Hit Key Points: {result.hit_key_points}")
            print(f"  Missing Key Points: {result.missing_key_points}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

def analyze_antonym_detection():
    """Test the Smart Detector on some examples"""
    
    print(f"\n🤖 TESTING SMART DETECTOR ON EVALUATION EXAMPLES")
    print("=" * 80)
    
    # Initialize Smart Detector
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    
    from services.smart_antonym_detector import SmartAntonymDetector
    smart_detector = SmartAntonymDetector(embedding_service, openai_client)
    
    # Test cases from the evaluation data
    test_cases = [
        # From Question 1 (Erosion)
        ("Erosion removes soil and rock materials.", "Erosion only happens in deserts where there's no plants to hold the soil together.", "Explain the role of erosion in shaping landscapes."),
        
        # From Question 3 (Plants)
        ("Use sunlight as energy", "Plants get their food from the soil by absorbing nutrients through their roots like animals eating.", "How do plants make their own food?"),
        
        # From Question 4 (Inflation)
        ("General increase in prices", "Inflation is when the government prints more money and everyone gets richer.", "Explain inflation"),
    ]
    
    for i, (key_point, student_answer, question) in enumerate(test_cases):
        print(f"\n  Test Case {i+1}:")
        print(f"  Key Point: '{key_point}'")
        print(f"  Student Answer: '{student_answer}'")
        print(f"  Question: '{question}'")
        
        try:
            result = smart_detector.detect_antonyms(key_point, student_answer, question)
            print(f"  Is Antonym: {result.is_antonym}")
            print(f"  Confidence: {result.confidence.value}")
            print(f"  Method: {result.method}")
            print(f"  Evidence: {result.evidence}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    analyze_evaluation_examples()
    analyze_antonym_detection()
