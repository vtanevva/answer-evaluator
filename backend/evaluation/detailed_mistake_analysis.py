"""
Detailed mistake analysis - shows exactly which answers were correct/incorrect
with actual scores and reasoning
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

def analyze_detailed_mistakes():
    """Analyze each answer in detail to show exact mistakes"""
    
    print("🔍 DETAILED MISTAKE ANALYSIS")
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
    
    # Analyze each question in detail
    for question in questions[:3]:  # First 3 questions only for detailed analysis
        analyze_question_detailed(grading_service, question)

def analyze_question_detailed(grading_service, question):
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
            result = grading_service.grade_answer(question_id, answer)
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
            result = grading_service.grade_answer(question_id, answer)
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

def analyze_antonym_detection_impact():
    """Analyze how antonym detection affects the results"""
    
    print(f"\n🤖 ANTONYM DETECTION IMPACT ANALYSIS")
    print("=" * 80)
    
    # Initialize Smart Detector
    api_key = os.getenv("OPENAI_API_KEY")
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    
    from services.smart_antonym_detector import SmartAntonymDetector
    smart_detector = SmartAntonymDetector(embedding_service, openai_client)
    
    # Load the evaluation data
    with open('annotated_questions.json', 'r') as f:
        questions = json.load(f)
    
    # Test cases that should be caught by antonym detection
    test_cases = [
        # Question 1 (Erosion) - Correct vs Incorrect
        ("Erosion removes soil and rock materials.", "Erosion only happens in deserts where there's no plants to hold the soil together.", "Explain the role of erosion in shaping landscapes."),
        ("Water, wind, and ice are erosion agents.", "Erosion is caused by earthquakes that break up the ground and make it loose.", "Explain the role of erosion in shaping landscapes."),
        
        # Question 3 (Plants) - Correct vs Incorrect  
        ("Use sunlight as energy", "Plants get their food from the soil by absorbing nutrients through their roots like animals eating.", "How do plants make their own food?"),
        ("Take in carbon dioxide", "Plants eat insects and small bugs that get stuck on their leaves to get protein.", "How do plants make their own food?"),
        
        # Question 4 (Inflation) - Correct vs Incorrect
        ("General increase in prices", "Inflation is when the government prints more money and everyone gets richer.", "Explain inflation"),
        ("Reduction of purchasing power", "Inflation only affects luxury items, basic necessities stay the same price.", "Explain inflation"),
    ]
    
    print(f"Testing {len(test_cases)} cases for antonym detection:")
    
    antonym_detected_count = 0
    for i, (key_point, student_answer, question) in enumerate(test_cases):
        print(f"\n  Test Case {i+1}:")
        print(f"  Key Point: '{key_point}'")
        print(f"  Student Answer: '{student_answer}'")
        
        try:
            result = smart_detector.detect_antonyms(key_point, student_answer, question)
            print(f"  Is Antonym: {result.is_antonym}")
            print(f"  Confidence: {result.confidence.value}")
            print(f"  Method: {result.method}")
            
            if result.is_antonym:
                antonym_detected_count += 1
                print(f"  ✅ CONTRADICTION DETECTED!")
            else:
                print(f"  ❌ Contradiction NOT detected")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📊 ANTONYM DETECTION SUMMARY:")
    print(f"  Contradictions Detected: {antonym_detected_count}/{len(test_cases)} ({antonym_detected_count/len(test_cases)*100:.1f}%)")
    
    if antonym_detected_count == 0:
        print(f"  🚨 ISSUE: Smart Detector not catching obvious contradictions!")
        print(f"  This explains why incorrect answers are getting high scores.")
    elif antonym_detected_count < len(test_cases) / 2:
        print(f"  ⚠️  Smart Detector needs improvement to catch more contradictions.")
    else:
        print(f"  ✅ Smart Detector is working well!")

if __name__ == "__main__":
    analyze_detailed_mistakes()
    analyze_antonym_detection_impact()



