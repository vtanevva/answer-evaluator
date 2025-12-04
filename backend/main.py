import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from openai import OpenAI

from core.config import settings
from routes.routes import router
from routes import routes
from services.question_service import QuestionService
from services.grading_service import GradingService

# Conditionally import NLI service only if needed
if getattr(settings.grading, 'grading_method', 'embedding') in ['nli', 'hybrid']:
    from services.nli_grading_service import NLIGradingService

# Load environment variables
load_dotenv()

# Global service instances
question_service: QuestionService = None
grading_service = None
_initialized = False


def initialize_services():
    """Initialize services on first request (Windows-compatible)"""
    global question_service, grading_service, _initialized
    
    if _initialized:
        return
    
    print("Starting Answer Evaluator Backend...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables!")

    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise RuntimeError("PINECONE_API_KEY not found in environment variables!")

    openai_client = OpenAI(api_key=api_key)
    print("OpenAI API key loaded")
    print("Pinecone API key loaded")

    question_service = QuestionService()
    question_service.load_questions_bank()

    grading_method = getattr(settings.grading, 'grading_method', 'embedding')
    
    if grading_method == 'nli':
        print("Using NLI-based grading (semantic inference)")
        grading_service = NLIGradingService(question_service)
    elif grading_method == 'hybrid':
        print("Using HYBRID grading (embedding + NLI verification)")
        print("  ✓ High similarity (>=85%): Auto-pass (no NLI check)")
        print("  ✓ Mid similarity (70-85%): NLI verification (contradiction-aware)")
        print("  ✓ Low similarity (<70%): Embedding fallback (NLI deep check disabled)")
        grading_service = GradingService(question_service, openai_client)
        grading_service.precompute_embeddings()
    else:
        print("Using embedding-based grading (cosine similarity)")
        grading_service = GradingService(question_service, openai_client)
        grading_service.precompute_embeddings()

    # Inject into routes module
    routes.question_service = question_service
    routes.grading_service = grading_service

    _initialized = True
    print("Backend ready!")


# Create FastAPI app WITHOUT lifespan (Windows compatibility)
app = FastAPI(
    title=settings.server.title,
    description=settings.server.description
)

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    """Initialize all services when the server starts"""
    print("\n🚀 Starting Answer Evaluator Backend...")
    initialize_services()
    print("✅ Server ready to accept requests!\n")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.allowed_origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=settings.cors.allowed_methods,
    allow_headers=settings.cors.allowed_headers,
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host=settings.server.host,
            port=settings.server.port,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user. Goodbye!")
    except Exception as e:
        print(f"\nServer crashed: {e}")