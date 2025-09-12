#!/usr/bin/env python3
"""
Test script to verify the synthetic data generator works
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if API key is properly loaded"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"✅ API Key found: {api_key[:10]}...{api_key[-4:]}")
        return True
    else:
        print("❌ No API key found in environment variables")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    try:
        import openai
        print("✅ OpenAI module imported successfully")
        
        # Test both old and new versions
        try:
            from openai import OpenAI
            print("✅ New OpenAI client available")
        except ImportError:
            print("✅ Using legacy OpenAI client")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Synthetic Data Generator Setup")
    print("=" * 50)
    
    # Test API key
    api_ok = test_api_key()
    
    # Test imports
    imports_ok = test_imports()
    
    print("\n" + "=" * 50)
    if api_ok and imports_ok:
        print("🎉 All tests passed! Ready to run the generator.")
        print("\nTo run the generator:")
        print("python synthetic_data_generator_enhanced.py")
    else:
        print("❌ Some tests failed. Please check your setup.")

if __name__ == "__main__":
    main()
