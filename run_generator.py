#!/usr/bin/env python3
"""
Quick setup and run script for the synthetic data generator
"""

import os
import subprocess
import sys

def check_api_key():
    """Check if OpenAI API key is set"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("\nTo set your API key:")
        print("1. Get your API key from: https://platform.openai.com/api-keys")
        print("2. Set it as an environment variable:")
        print("   Windows: set OPENAI_API_KEY=your-key-here")
        print("   Mac/Linux: export OPENAI_API_KEY=your-key-here")
        print("   Or create a .env file with: OPENAI_API_KEY=your-key-here")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def main():
    """Main setup and run function"""
    print("🚀 Synthetic Data Generator Setup")
    print("=" * 40)
    
    # Check API key
    if not check_api_key():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n🎯 Starting data generation...")
    print("=" * 40)
    
    # Run the main generator
    try:
        subprocess.check_call([sys.executable, "synthetic_data_generator.py"])
    except subprocess.CalledProcessError:
        print("❌ Data generation failed")
    except KeyboardInterrupt:
        print("\n⏹️ Generation interrupted by user")

if __name__ == "__main__":
    main()
