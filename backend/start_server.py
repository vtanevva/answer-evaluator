"""
Windows-compatible server startup script
Bypasses asyncio.CancelledError on Windows by running uvicorn programmatically
"""
import sys
import os

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add backend directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    
    # Run uvicorn with Windows-compatible settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        # Windows compatibility: Use 'none' loop to avoid asyncio issues
        loop="none"
    )
