"""
Simple accuracy test - Run this to see the improvement
"""

import os
import sys
from dotenv import load_dotenv

# Add the backend directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from parent directory (.env file)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

def test_accuracy():
    """Simple test to show accuracy improvement"""
    
    print("🧪 ACCURACY TEST")
    print("=" * 50)
    
    # Check if OpenAI key exists
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("Please set your OpenAI API key in .env file")
        return
    
    print("✅ OpenAI API key found")
    
    try:
        # Import and test Smart Detector
        import openai
        from services.smart_antonym_detector import SmartAntonymDetector
        from services.embedding_service import EmbeddingService
        
        print("✅ Smart Detector imported successfully")
        
        # Initialize
        openai_client = openai.OpenAI(api_key=api_key)
        embedding_service = EmbeddingService(openai_client)
        smart_detector = SmartAntonymDetector(embedding_service, openai_client)
        
        print("✅ Smart Detector initialized")
        
        # Test cases that were failing before
        failing_cases = [
            ("I like this", "I not like this"),
            ("I like this", "I don't like this"),
            ("This is good", "This is not good"),
            ("I agree", "I not agree"),
        ]
        
        # Test cases that were working before
        working_cases = [
            ("increase", "decrease"),
            ("I like this", "I dislike this"),
            ("good", "bad"),
        ]
        
        print("\n🔴 TESTING PREVIOUSLY FAILING CASES:")
        print("-" * 40)
        failing_solved = 0
        
        for text1, text2 in failing_cases:
            try:
                result = smart_detector.detect_antonyms(text1, text2)
                is_antonym = result.is_antonym
                if is_antonym:
                    failing_solved += 1
                
                status = "✅ SOLVED" if is_antonym else "❌ STILL FAILING"
                print(f"{status} '{text1}' vs '{text2}'")
                print(f"    Method: {result.method}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print(f"\n🟢 TESTING PREVIOUSLY WORKING CASES:")
        print("-" * 40)
        working_maintained = 0
        
        for text1, text2 in working_cases:
            try:
                result = smart_detector.detect_antonyms(text1, text2)
                is_antonym = result.is_antonym
                if is_antonym:
                    working_maintained += 1
                
                status = "✅ STILL WORKS" if is_antonym else "❌ BROKEN"
                print(f"{status} '{text1}' vs '{text2}'")
                print(f"    Method: {result.method}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Calculate accuracy
        total_tests = len(failing_cases) + len(working_cases)
        total_correct = failing_solved + working_maintained
        accuracy = (total_correct / total_tests) * 100
        
        print(f"\n📊 RESULTS:")
        print("=" * 50)
        print(f"🔴 Previously Failing Cases: {failing_solved}/{len(failing_cases)} solved")
        print(f"🟢 Previously Working Cases: {working_maintained}/{len(working_cases)} maintained")
        print(f"🎯 Overall Accuracy: {total_correct}/{total_tests} ({accuracy:.1f}%)")
        
        if accuracy >= 85:
            print(f"\n✅ SUCCESS! Smart Detector is working!")
            print(f"   Accuracy improved significantly!")
        else:
            print(f"\n⚠️  Smart Detector needs optimization")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_accuracy()
