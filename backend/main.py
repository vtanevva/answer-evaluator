import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
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
    description=settings.server.description,
    version="1.0.0",
    docs_url="/docs",           # Swagger UI
    redoc_url="/redoc",         # ReDoc alternative documentation
    openapi_url="/openapi.json" # OpenAPI schema
)


def custom_openapi():
    """Generate custom OpenAPI schema with enhanced documentation"""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.server.title,
        version="1.0.0",
        description="""
## Answer Evaluator API

A powerful REST API for grading student answers using semantic embeddings and natural language inference.

### Key Features

✅ **Semantic Similarity Grading**: Uses embeddings to understand answer meaning
✅ **NLI Verification**: Natural Language Inference for contradiction detection
✅ **Hybrid Evaluation**: Combines multiple AI techniques for high accuracy
✅ **Answer Caching**: 10-200x faster grading for similar answers
✅ **Rubric-Based**: Compare answers against customizable key points
✅ **Multi-Language**: Supports economics, biology, geography, and more

### How It Works

1. **Student submits answer** to a question
2. **System validates** the answer content
3. **Embeddings generated** for semantic understanding
4. **Cache checked** for identical/similar previous answers
5. **Key points compared** using semantic similarity
6. **Score calculated** based on coverage
7. **Result cached** for future optimization

### Grading Pipeline

```
Submit Answer
    ↓
Check Cache (hit → instant result)
    ↓
Validate Answer (length, content)
    ↓
Generate Embeddings
    ↓
Compare with Key Points
    ↓
Verify with NLI (if enabled)
    ↓
Calculate Score
    ↓
Cache Result
    ↓
Return Grade
```

### Configuration

- **Grading Method**: Embedding, NLI, or Hybrid (default)
- **Models**: GTE-Multilingual local, or OpenAI API
- **Similarity Threshold**: 0.99 for caching (configurable)
- **NLI Models**: DeBERTa small (fast) or base (accurate)

### Getting Started

1. **Get a random question**: `GET /question`
2. **Submit an answer**: `POST /answer`
3. **Review the grade**: Score, key points, feedback
4. **Check cache stats**: `GET /cache/stats`

### Example Request

```json
POST /answer
{
  "question_id": 1,
  "user_answer": "Inflation is a general increase in prices over time"
}
```

### Example Response

```json
{
  "score": 100.0,
  "hit_key_points": [
    "General increase in prices",
    "Reduction of purchasing power"
  ],
  "missing_key_points": [],
  "feedback": "Correct! You covered all the key points."
}
```

        """,
        routes=app.routes,
        # tags_metadata=[
        #     {
        #         "name": "Health",
        #         "description": "API health and status endpoints"
        #     },
        #     {
        #         "name": "Questions",
        #         "description": "Retrieve questions from the question bank"
        #     },
        #     {
        #         "name": "Grading",
        #         "description": "Grade student answers using semantic analysis"
        #     },
        #     {
        #         "name": "Cache Management",
        #         "description": "Manage the answer cache for performance optimization"
        #     }
        # ],
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ]
    )
    
    # Add security info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
        "altText": "FastAPI Logo"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Set custom OpenAPI schema
app.openapi = custom_openapi

# Initialize services at startup
@app.on_event("startup")
async def startup_event():
    """Initialize all services when the server starts"""
    print("\n🚀 Starting Answer Evaluator Backend...")
    initialize_services()
    print("✅ Server ready to accept requests!\n")
    print("📖 API Documentation available at:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("   - OpenAPI Schema: http://localhost:8000/openapi.json\n")

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