"""
Test script to demonstrate the improved AI-powered antonym detection system
"""

import os
import sys
from dotenv import load_dotenv
import openai

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.ai_antonym_detector import AIAntonymDetector, AntonymConfidence
from services.embedding_service import EmbeddingService
from core.config import settings

# Load environment variables
load_dotenv()

def test_antonym_detection():
    """Test the AI-powered antonym detection with various examples"""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    antonym_detector = AIAntonymDetector(embedding_service, openai_client)
    
    print("🧪 Testing AI-Powered Antonym Detection System")
    print("=" * 60)
    
    # Test cases with expected results
    test_cases = [
        # Clear antonyms
        ("increase", "decrease", "The economy will", True),
        ("good", "bad", "The weather is", True),
        ("high", "low", "The temperature is", True),
        ("active", "inactive", "The system is", True),
        ("agree", "disagree", "I", True),
        
        # Non-antonyms (similar concepts)
        ("increase", "rise", "The price will", False),
        ("good", "excellent", "The service is", False),
        ("high", "tall", "The building is", False),
        
        # Context-dependent cases
        ("hot", "cold", "The water is", True),
        ("hot", "warm", "The water is", False),
        ("fast", "slow", "The car is", True),
        ("fast", "quick", "The car is", False),
        
        # Complex cases
        ("advantage", "disadvantage", "The main", True),
        ("benefit", "harm", "This will", True),
        ("positive", "negative", "The effect is", True),
        
        # Edge cases
        ("same", "different", "They are", True),
        ("similar", "different", "They are", True),
        ("include", "exclude", "We should", True),
    ]
    
    print(f"Running {len(test_cases)} test cases...\n")
    
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, (text1, text2, context, expected_is_antonym) in enumerate(test_cases, 1):
        print(f"Test {i:2d}: '{text1}' vs '{text2}' (context: '{context}')")
        
        try:
            result = antonym_detector.detect_antonyms(text1, text2, context)
            
            # Determine if prediction is correct
            prediction_correct = result.is_antonym == expected_is_antonym
            if prediction_correct:
                correct_predictions += 1
            
            # Display result
            status = "✅" if prediction_correct else "❌"
            expected_str = "antonym" if expected_is_antonym else "not antonym"
            actual_str = "antonym" if result.is_antonym else "not antonym"
            
            print(f"  {status} Expected: {expected_str}, Got: {actual_str}")
            print(f"     Confidence: {result.confidence.value}")
            print(f"     Method: {result.method}")
            print(f"     Evidence: {result.evidence}")
            if result.semantic_distance > 0:
                print(f"     Semantic Distance: {result.semantic_distance:.3f}")
            if result.context_similarity > 0:
                print(f"     Context Similarity: {result.context_similarity:.3f}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
    
    # Summary
    accuracy = (correct_predictions / total_tests) * 100
    print("=" * 60)
    print(f"📊 Test Results Summary:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Correct Predictions: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1f}%")
    
    if accuracy >= 80:
        print("🎉 Excellent performance!")
    elif accuracy >= 70:
        print("👍 Good performance!")
    elif accuracy >= 60:
        print("⚠️ Moderate performance - consider tuning parameters")
    else:
        print("❌ Poor performance - needs improvement")


def test_comparison_with_old_system():
    """Compare new AI system with old rule-based system"""
    
    print("\n" + "=" * 60)
    print("🔄 Comparing AI System vs Old Rule-Based System")
    print("=" * 60)
    
    # Initialize services
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    ai_detector = AIAntonymDetector(embedding_service, openai_client)
    
    # Import old system
    from services.text_processing import TextProcessor
    old_processor = TextProcessor()
    
    # Test cases that the old system might miss
    challenging_cases = [
        ("prosperous", "impoverished", "The country is"),
        ("efficient", "inefficient", "The process is"),
        ("reliable", "unreliable", "The system is"),
        ("sustainable", "unsustainable", "The practice is"),
        ("transparent", "opaque", "The process is"),
        ("flexible", "rigid", "The approach is"),
        ("stable", "volatile", "The market is"),
        ("secure", "vulnerable", "The system is"),
    ]
    
    print("Testing challenging cases that old system might miss:\n")
    
    for text1, text2, context in challenging_cases:
        print(f"Case: '{text1}' vs '{text2}' (context: '{context}')")
        
        # Test with AI system
        try:
            ai_result = ai_detector.detect_antonyms(text1, text2, context)
            print(f"  🤖 AI System: {ai_result.is_antonym} (confidence: {ai_result.confidence.value})")
        except Exception as e:
            print(f"  🤖 AI System: Error - {e}")
        
        # Test with old system
        try:
            tokens1 = old_processor.normalize_text(text1)
            tokens2 = old_processor.normalize_text(text2)
            old_result = (old_processor.has_polarity_conflict(tokens1, tokens2) or
                         old_processor.has_direction_conflict(tokens1, tokens2))
            print(f"  📜 Old System: {old_result}")
        except Exception as e:
            print(f"  📜 Old System: Error - {e}")
        
        print()


if __name__ == "__main__":
    test_antonym_detection()
    test_comparison_with_old_system()
