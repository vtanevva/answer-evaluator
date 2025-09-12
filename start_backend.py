#!/usr/bin/env python3
"""
Simple backend starter without reload to avoid warnings
"""

import uvicorn
import os
import sys

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

if __name__ == "__main__":
    print("🚀 Starting Answer Evaluator Backend...")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop")
    
    # Run without reload to avoid warnings
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=False
    )
