#!/usr/bin/env python3
"""
Frontend runner script with environment setup
"""

import os
import sys
import subprocess
import json

def check_node_version():
    """Check if Node.js is installed and version is compatible"""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Node.js not found")
            print("Please install Node.js 16+ from https://nodejs.org/")
            return False
        
        version = result.stdout.strip()
        print(f"✅ Node.js version: {version}")
        return True
    except FileNotFoundError:
        print("❌ Node.js not found")
        print("Please install Node.js 16+ from https://nodejs.org/")
        return False

def check_npm():
    """Check if npm is available"""
    try:
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ npm not found")
            return False
        
        version = result.stdout.strip()
        print(f"✅ npm version: {version}")
        return True
    except FileNotFoundError:
        print("❌ npm not found")
        return False

def check_dependencies():
    """Check if frontend dependencies are installed"""
    if not os.path.exists("frontend/node_modules"):
        print("❌ Frontend dependencies not installed")
        print("Run: cd frontend && npm install")
        return False
    
    print("✅ Frontend dependencies installed")
    return True

def check_backend_running():
    """Check if backend is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            print("✅ Backend is running")
            return True
    except:
        pass
    
    print("⚠️  Backend not detected on localhost:8000")
    print("Make sure to start the backend first with: python run_backend.py")
    return False

def main():
    """Main runner function"""
    print("🚀 Starting Answer Evaluator Frontend...")
    print("=" * 50)
    
    # Check prerequisites
    if not check_node_version():
        return
    
    if not check_npm():
        return
    
    if not check_dependencies():
        return
    
    check_backend_running()
    
    print("\n✅ All checks passed!")
    print("🌐 Starting frontend on http://localhost:3000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    
    # Change to frontend directory and run
    os.chdir("frontend")
    subprocess.run(["npm", "start"])

if __name__ == "__main__":
    main()
