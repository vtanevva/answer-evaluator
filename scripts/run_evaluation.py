#!/usr/bin/env python3
"""
Runner script for the evaluation benchmark with proper environment setup
"""

import os
import sys
from pathlib import Path

def main():
    """Main function to run the evaluation with proper setup"""
    
    # Get the project root directory (one level up from scripts)
    project_root = Path(__file__).parent.parent
    backend_dir = project_root / "backend"
    evaluation_dir = backend_dir / "evaluation"
    
    print(f"🔍 Project root: {project_root.absolute()}")
    print(f"🔍 Backend dir: {backend_dir.absolute()}")
    print(f"🔍 Evaluation dir: {evaluation_dir.absolute()}")
    
    # Add backend directory to Python path for imports
    sys.path.insert(0, str(backend_dir))
    
    # Try to load environment variables from .env file
    try:
        from dotenv import load_dotenv
        
        # Try loading from backend/.env first, then root .env
        backend_env = backend_dir / ".env"
        root_env = project_root / ".env"
        
        if backend_env.exists():
            load_dotenv(backend_env)
            print(f"🔧 Loaded environment from {backend_env}")
        elif root_env.exists():
            load_dotenv(root_env)
            print(f"🔧 Loaded environment from {root_env}")
        else:
            load_dotenv()  # Load from current directory if available
            
    except ImportError:
        print("⚠️ python-dotenv not installed, using system environment only")
    
    # Change to backend directory for relative imports
    original_cwd = os.getcwd()
    os.chdir(backend_dir)
    
    try:
        # Import and run the benchmark from the evaluation subdirectory
        sys.path.insert(0, str(evaluation_dir))
        from evaluation_script import main as benchmark_main
        benchmark_main()
    finally:
        # Restore original directory
        os.chdir(original_cwd)

if __name__ == "__main__":
    main()
