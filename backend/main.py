"""
Answer Evaluator Backend - FastAPI Application
Implements embedding-based question-answer evaluation using OpenAI embeddings
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import numpy as np
import openai
import json
import os
from dotenv import load_dotenv
import uvicorn
from load_questions import load_questions_from_file

# Load environment variables from .env file
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Answer Evaluator", description="Evaluate student answers using embeddings")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for questions and embeddings
questions_bank = []
key_point_embeddings = {}  # {question_id: [embedding1, embedding2, ...]}
questions_by_id = {}  # {question_id: question_data}
key_point_keywords: Dict[int, List[set]] = {}  # {question_id: [set(tokens), ...]}

# Pydantic models for request/response
class QuestionResponse(BaseModel):
    question_id: int
    question_text: str

class AnswerRequest(BaseModel):
    question_id: int
    user_answer: str

class AnswerResponse(BaseModel):
    score: float
    hit_key_points: List[str]
    missing_key_points: List[str]
    feedback: str

def load_questions_bank():
    """
    Load questions from backend/questions.json if present; fallback to built-in sample.
    """
    global questions_bank, questions_by_id
    
    # Try to load from file first
    file_path = os.path.join(os.path.dirname(__file__), "questions.json")
    loaded = load_questions_from_file(file_path)
    if loaded:
        questions_bank = loaded
        print(f"✅ Loaded {len(questions_bank)} questions from {file_path}")
    else:
        # Fallback sample (3 questions)
        questions_bank = [
            {
                "question_id": 1,
                "question_text": "What is inflation?",
                "key_points": [
                    {"text": "General increase in prices", "weight": 1},
                    {"text": "Reduction of purchasing power", "weight": 1}
                ]
            },
            {
                "question_id": 2,
                "question_text": "Explain supply and demand",
                "key_points": [
                    {"text": "Relationship between price and quantity demanded", "weight": 1},
                    {"text": "As price increases, demand decreases", "weight": 1}
                ]
            },
            {
                "question_id": 3,
                "question_text": "What are three adaptations that help desert animals survive?",
                "key_points": [
                    {"text": "Store water in their bodies", "weight": 1},
                    {"text": "Active at night", "weight": 1},
                    {"text": "Light-colored skin", "weight": 1}
                ]
            }
        ]
        print(f"⚠️ Falling back to built-in sample. Loaded {len(questions_bank)} questions")
    
    # Create lookup dictionary for faster access
    questions_by_id = {q["question_id"]: q for q in questions_bank}

def get_embedding(text: str) -> List[float]:
    """
    Get embedding for a text using OpenAI text-embedding-ada-002
    
    This is where the magic happens:
    1. Send text to OpenAI embedding API
    2. Get back a 1536-dimensional vector
    3. This vector represents the semantic meaning of the text
    
    Args:
        text: The text to embed
        
    Returns:
        List of 1536 float values representing the embedding
    """
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to get embedding")

def compute_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors
    
    Cosine similarity measures the angle between two vectors:
    - 1.0 = identical direction (perfect match)
    - 0.0 = perpendicular (no similarity)
    - -1.0 = opposite direction (completely different)
    
    Formula: cos(θ) = (A · B) / (||A|| × ||B||)
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Similarity score between -1 and 1
    """
    # Convert to numpy arrays for efficient computation
    a = np.array(vec1)
    b = np.array(vec2)
    
    # Compute dot product
    dot_product = np.dot(a, b)
    
    # Compute magnitudes
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # Avoid division by zero
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # Return cosine similarity
    return dot_product / (norm_a * norm_b)

# ----------------------------
# Lightweight lexical matching
# ----------------------------
_STOPWORDS = set(
    "the a an and or of in on at to for from as by is are was were be being been it its this that those these with without into over under which who whom whose what when where how why can could should would may might must do does did doing done have has had having not no nor if then else than also more most much many few several such same other another vs versus per each both either neither all any some most mostly mainly typically usually about around roughly approximately like include including e.g. i.e.".split()
)

def _normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t and t not in _STOPWORDS]
    # Naive stemming: strip common suffixes
    stemmed = []
    for t in tokens:
        for suf in ("ing", "ed", "es", "s"):
            if len(t) > 4 and t.endswith(suf):
                t = t[: -len(suf)]
                break
        stemmed.append(t)
    return stemmed

def _token_overlap(user_tokens: List[str], kp_tokens: List[str]) -> float:
    us = set(user_tokens)
    ks = set(kp_tokens)
    if not ks:
        return 0.0
    return len(us & ks) / len(ks)

def precompute_embeddings():
    """
    Precompute embeddings for all key points at startup
    
    This is where we convert all key points to embeddings once:
    1. Loop through all questions
    2. For each key point, get its embedding
    3. Store embeddings in memory for fast lookup
    
    In production with a vector DB, this would be done once and stored in the DB
    """
    global key_point_embeddings, key_point_keywords
    
    print("🔄 Precomputing embeddings for key points...")
    
    for question in questions_bank:
        question_id = question["question_id"]
        embeddings = []
        keywords_list: List[set] = []
        
        for key_point in question["key_points"]:
            text = key_point["text"]
            embedding = get_embedding(text)
            embeddings.append(embedding)
            keywords_list.append(set(_normalize(text)))
            print(f"  ✅ Embedded: '{text[:30]}...'")
        
        key_point_embeddings[question_id] = embeddings
        key_point_keywords[question_id] = keywords_list
        print(f"  📝 Question {question_id}: {len(embeddings)} key points embedded")
    
    print(f"✅ Precomputed embeddings for {len(questions_bank)} questions")

def evaluate_answer(question_id: int, user_answer: str) -> AnswerResponse:
    """
    Evaluate user answer against key points using embedding similarity
    
    This is the core evaluation logic:
    1. Get user answer embedding
    2. Compare with each key point embedding using cosine similarity
    3. Mark key points as "hit" or "missing" based on similarity threshold
    4. Calculate score and generate feedback
    
    Args:
        question_id: ID of the question being answered
        user_answer: The student's answer text
        
    Returns:
        AnswerResponse with score, hit/missing points, and feedback
    """
    # Get question data
    if question_id not in questions_by_id:
        raise HTTPException(status_code=404, detail="Question not found")
    
    question = questions_by_id[question_id]
    key_points = question["key_points"]
    
    # Split user answer into sentences to reduce noise and compute
    # sentence-level embeddings (a short answer may only match one sentence).
    # Simple split on punctuation; for production consider spaCy.
    raw_sentences = re.split(r"[\.!?\n]+", user_answer)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        sentences = [user_answer]

    # Batch embed all sentences in one call to reduce cost/latency
    try:
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=sentences
        )
        sent_embeddings = [d.embedding for d in resp.data]
    except Exception:
        # Fallback to single embedding of full answer
        sent_embeddings = [get_embedding(user_answer)]

    user_tokens = _normalize(user_answer)
    
    # Compare with each key point embedding
    hit_key_points = []
    missing_key_points = []
    # Stricter thresholds: require higher semantic similarity,
    # or moderate similarity plus sufficient lexical overlap.
    high_sim = 0.88
    mid_sim = 0.83
    min_overlap = 0.5  # at least half of kp tokens detected
    
    for i, key_point in enumerate(key_points):
        key_point_embedding = key_point_embeddings[question_id][i]
        kp_tokens = list(key_point_keywords[question_id][i]) if key_point_keywords.get(question_id) else _normalize(key_point["text"])

        # Max sentence-level cosine similarity against this key point
        similarity = max(
            (compute_cosine_similarity(se, key_point_embedding) for se in sent_embeddings),
            default=0.0,
        )
        # Lexical overlap between user's tokens and KP tokens
        overlap = _token_overlap(user_tokens, kp_tokens)

        is_hit = similarity >= high_sim or (similarity >= mid_sim and overlap >= min_overlap)

        if is_hit:
            hit_key_points.append(key_point["text"])
        else:
            missing_key_points.append(key_point["text"])
        
        print(
            f"  📊 Key point '{key_point['text'][:30]}...': sim={similarity:.3f}, overlap={overlap:.2f}, hit={is_hit}"
        )
    
    # Calculate score
    total_points = len(key_points)
    hit_count = len(hit_key_points)
    score = (hit_count / total_points) * 100 if total_points > 0 else 0
    
    # Generate feedback
    if score == 100:
        feedback = "Correct! You covered all the key points."
    elif score >= 50:
        feedback = f"Partial - missing {len(missing_key_points)} key point(s). Good start!"
    else:
        feedback = "Incorrect - try again. Review the material and provide a more complete answer."
    
    return AnswerResponse(
        score=round(score, 1),
        hit_key_points=hit_key_points,
        missing_key_points=missing_key_points,
        feedback=feedback
    )

@app.on_event("startup")
async def startup_event():
    """
    Startup event - load questions and precompute embeddings
    This runs once when the server starts
    """
    print("🚀 Starting Answer Evaluator Backend...")
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key in .env file")
        return
    
    # Set OpenAI API key
    openai.api_key = api_key
    print("✅ OpenAI API key loaded")
    
    # Load questions and precompute embeddings
    load_questions_bank()
    precompute_embeddings()
    
    print("✅ Backend ready!")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Answer Evaluator API is running", "questions_loaded": len(questions_bank)}

@app.get("/question", response_model=QuestionResponse)
async def get_random_question():
    """
    Get a random question from the question bank
    
    Returns:
        QuestionResponse with question_id and question_text
    """
    import random
    
    if not questions_bank:
        raise HTTPException(status_code=500, detail="No questions loaded")
    
    # Pick a random question
    question = random.choice(questions_bank)
    
    return QuestionResponse(
        question_id=question["question_id"],
        question_text=question["question_text"]
    )

@app.post("/answer", response_model=AnswerResponse)
async def evaluate_answer_endpoint(request: AnswerRequest):
    """
    Evaluate a user's answer against the question's key points
    
    This endpoint:
    1. Checks if the answer is valid (not empty or "I don't know")
    2. Gets embedding for user answer
    3. Compares with precomputed key point embeddings
    4. Returns evaluation results
    
    Args:
        request: AnswerRequest with question_id and user_answer
        
    Returns:
        AnswerResponse with score, hit/missing points, and feedback
    """
    # Handle empty, too short, or "I don't know" answers
    user_answer_clean = request.user_answer.strip()
    
    if not user_answer_clean or user_answer_clean.lower() in ["i don't know", "i don't know.", "dont know"]:
        return AnswerResponse(
            score=0.0,
            hit_key_points=[],
            missing_key_points=[],
            feedback="Please try again. Even if you're unsure, try to explain what you think might be the answer."
        )
    
    # Reject very short answers (less than 10 characters or single words)
    if len(user_answer_clean) < 10 or len(user_answer_clean.split()) < 2:
        return AnswerResponse(
            score=0.0,
            hit_key_points=[],
            missing_key_points=[],
            feedback="Your answer is too short. Please provide a more detailed explanation with at least a few words."
        )
    
    # Evaluate the answer
    return evaluate_answer(request.question_id, request.user_answer)

@app.get("/questions")
async def get_all_questions():
    """
    Get all questions (for debugging/development)
    """
    return {"questions": questions_bank}

if __name__ == "__main__":
    """
    Run the FastAPI server
    Development server - use uvicorn for production
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
