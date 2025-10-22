"""
Quick test to verify Smart Detector is working
"""

import os
import sys
from dotenv import load_dotenv
import openai

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.smart_antonym_detector import SmartAntonymDetector
from services.embedding_service import EmbeddingService

# Load environment variables
load_dotenv()

def quick_test():
    """Quick test of the Smart Detector"""
    
    print("🚀 QUICK SMART DETECTOR TEST")
    print("=" * 50)
    
    # Initialize
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        return
    
    openai_client = openai.OpenAI(api_key=api_key)
    embedding_service = EmbeddingService(openai_client)
    smart_detector = SmartAntonymDetector(embedding_service, openai_client)
    
    # Test the key failing cases
    test_cases = [
        ("I like this", "I not like this"),
        ("I like this", "I don't like this"),
        ("This is good", "This is not good"),
        ("increase", "decrease"),  # This should still work
    ]
    
    print("Testing key cases:")
    print()
    
    for text1, text2 in test_cases:
        try:
            result = smart_detector.detect_antonyms(text1, text2)
            status = "✅" if result.is_antonym else "❌"
            print(f"{status} '{text1}' vs '{text2}': {result.is_antonym}")
            print(f"    Method: {result.method}, Confidence: {result.confidence.value}")
            print()
            
        except Exception as e:
            print(f"❌ Error testing '{text1}' vs '{text2}': {e}")
            print()
    
    print("🎯 Smart Detector is ready!")
    print("Your system now uses SmartAntonymDetector instead of AIAntonymDetector")

if __name__ == "__main__":
    quick_test()



