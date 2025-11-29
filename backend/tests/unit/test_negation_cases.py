"""
Test script to demonstrate how the advanced antonym detector handles negation cases
"""

import os
import sys
from dotenv import load_dotenv
import openai

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.advanced_antonym_detector import AdvancedAntonymDetector, AntonymConfidence
from services.embedding_service import EmbeddingService
from core.config import settings

# Load environment variables
load_dotenv()

def test_negation_cases():
    """Test negation and antonym cases that the current system struggles with"""
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    
    # Test with current system first
    print("🔍 Testing Current System (Basic AI Antonym Detector)")
    print("=" * 60)
    
    try:
        from services.ai_antonym_detector import AIAntonymDetector
        current_detector = AIAntonymDetector(embedding_service, openai_client)
        
        # Test cases that should be antonyms
        negation_cases = [
            ("I like this", "I not like this", "My opinion is"),
            ("I like this", "I dislike this", "My opinion is"),
            ("I like this", "I hate this", "My opinion is"),
            ("I agree", "I not agree", "My position is"),
            ("I agree", "I disagree", "My position is"),
            ("I support this", "I not support this", "My stance is"),
            ("I support this", "I oppose this", "My stance is"),
            ("This is good", "This is not good", "My assessment is"),
            ("This is good", "This is bad", "My assessment is"),
            ("This works", "This not works", "My experience is"),
            ("This works", "This fails", "My experience is"),
        ]
        
        print("Testing negation cases with current system:")
        for text1, text2, context in negation_cases:
            try:
                result = current_detector.detect_antonyms(text1, text2, context)
                status = "✅" if result.is_antonym else "❌"
                print(f"  {status} '{text1}' vs '{text2}': {result.is_antonym} (confidence: {result.confidence.value})")
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
    except Exception as e:
        print(f"❌ Current system error: {e}")
    
    print("\n" + "=" * 60)
    print("🚀 Testing Advanced System (Multi-Model Approach)")
    print("=" * 60)
    
    try:
        # Test with advanced system
        advanced_detector = AdvancedAntonymDetector(embedding_service, openai_client)
        
        print("Testing negation cases with advanced system:")
        for text1, text2, context in negation_cases:
            try:
                result = advanced_detector.detect_antonyms(text1, text2, context)
                status = "✅" if result.is_antonym else "❌"
                print(f"  {status} '{text1}' vs '{text2}': {result.is_antonym} (confidence: {result.confidence.value})")
                print(f"      Method: {result.method}")
                print(f"      Evidence: {result.evidence}")
                if result.model_scores:
                    print(f"      Model Scores: {result.model_scores}")
                print()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Advanced system error: {e}")
        import traceback
        traceback.print_exc()

def test_specific_negation_patterns():
    """Test specific negation patterns that should be detected"""
    
    print("\n" + "=" * 60)
    print("🧪 Testing Specific Negation Patterns")
    print("=" * 60)
    
    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    
    try:
        advanced_detector = AdvancedAntonymDetector(embedding_service, openai_client)
        
        # Test different types of negations
        test_cases = [
            # Direct negation with "not"
            ("I like pizza", "I do not like pizza", "My food preference is"),
            ("I like pizza", "I don't like pizza", "My food preference is"),
            ("I like pizza", "I dislike pizza", "My food preference is"),
            
            # Negation with prefixes
            ("I agree with you", "I disagree with you", "My opinion is"),
            ("I approve this", "I disapprove this", "My decision is"),
            ("I like this idea", "I dislike this idea", "My feeling is"),
            
            # Stronger antonyms
            ("I love this", "I hate this", "My feeling is"),
            ("I support this", "I oppose this", "My stance is"),
            ("I accept this", "I reject this", "My decision is"),
            
            # Complex cases
            ("This is a good solution", "This is not a good solution", "My assessment is"),
            ("This works well", "This does not work well", "My experience is"),
            ("I think this is correct", "I think this is incorrect", "My belief is"),
        ]
        
        correct_detections = 0
        total_tests = len(test_cases)
        
        for text1, text2, context in test_cases:
            try:
                result = advanced_detector.detect_antonyms(text1, text2, context)
                is_correct = result.is_antonym
                if is_correct:
                    correct_detections += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} '{text1}' vs '{text2}': {is_correct}")
                print(f"      Confidence: {result.confidence.value}, Method: {result.method}")
                print(f"      Evidence: {result.evidence}")
                print()
                
            except Exception as e:
                print(f"  ❌ Error testing '{text1}' vs '{text2}': {e}")
        
        accuracy = (correct_detections / total_tests) * 100
        print(f"📊 Negation Detection Accuracy: {accuracy:.1f}% ({correct_detections}/{total_tests})")
        
    except Exception as e:
        print(f"❌ Error in negation pattern testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_negation_cases()
    test_specific_negation_patterns()
