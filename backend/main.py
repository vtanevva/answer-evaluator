import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

from core.config import settings
from routes.routes import router
from routes import routes
from services.question_service import QuestionService
from services.grading_service import GradingService

# Load environment variables from .env file
load_dotenv()

# Global service instances
question_service: QuestionService = None
grading_service: GradingService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler - initialize services on startup and cleanup on shutdown
    This replaces the deprecated @app.on_event("startup") decorator
    """
    global question_service, grading_service
    
    # Startup
    print("🚀 Starting Answer Evaluator Backend...")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or len(api_key) == 0:
        print("❌ OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key in .env file")
        raise RuntimeError("OpenAI API key not found")
    
    # Check for Pinecone API key
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key or len(pinecone_key) == 0:
        print("❌ PINECONE_API_KEY not found in environment variables!")
        print("Please set your Pinecone API key in .env file")
        raise RuntimeError("Pinecone API key not found")
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=api_key)
    print("✅ OpenAI API key loaded")
    print("✅ Pinecone API key loaded")
    
    # Initialize services
    question_service = QuestionService()
    question_service.load_questions_bank()
    
    grading_service = GradingService(question_service, openai)
    grading_service.precompute_embeddings()
    
    # Set service instances in routes module for dependency injection
    routes.question_service = question_service
    routes.grading_service = grading_service
    
    print("✅ Backend ready!")
    
    yield  # Application runs here
    
    # Shutdown (cleanup if needed)
    print("🔄 Shutting down Answer Evaluator Backend...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.server.title,
    description=settings.server.description,
    lifespan=lifespan
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allowed_methods,
    allow_headers=settings.cors.allowed_headers,
)

# Include API routes
app.include_router(router)

if __name__ == "__main__":
    """
    Run the FastAPI server
    Development server - use uvicorn for production
    """
    uvicorn.run(
        "main:app",  # Use import string format for reload support
        host=settings.server.host, 
        port=settings.server.port, 
        reload=settings.server.reload
    )
