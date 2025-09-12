#!/usr/bin/env python3
"""
Backend runner script with environment setup
"""

import os
import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import openai
        import numpy
        import uvicorn
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r backend/requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has API key"""
    env_path = "backend/.env"
    if not os.path.exists(env_path):
        print("❌ .env file not found in backend directory")
        print("Copy backend/env_example.txt to backend/.env and add your OpenAI API key")
        return False
    
    # Only check presence of key, never hardcode or compare actual secrets
    with open(env_path, 'r') as f:
        content = f.read()
        if "OPENAI_API_KEY=" not in content or content.strip().endswith("your-openai-api-key-here"):
            print("❌ Please set your OpenAI API key in backend/.env")
            return False
    
    print("✅ .env file configured")
    return True

def main():
    """Main runner function"""
    print("🚀 Starting Answer Evaluator Backend...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_python_version():
        return
    
    if not check_dependencies():
        return
    
    if not check_env_file():
        return
    
    print("\n✅ All checks passed!")
    print("🌐 Starting server on http://localhost:8000")
    print("📚 API docs available at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Change to backend directory and run
    os.chdir("backend")
    subprocess.run([sys.executable, "main.py"])

if __name__ == "__main__":
    main()
