"""
API routes for the Answer Evaluator application
"""

import random
from fastapi import APIRouter, HTTPException, Depends

from models.models import (
    QuestionResponse,
    AnswerRequest,
    AnswerResponse,
    HealthCheckResponse,
    AllQuestionsResponse,
    AddQuestionRequest,
    AddQuestionResponse,
    DeleteQuestionResponse,
    BulkQuestionsFromTextRequest,
    BulkQuestionsFromTextResponse,
    Question,
)
from services.question_service import QuestionService
from services.grading_service import GradingService
from core.config import settings
import json
import os


# Create router instance
router = APIRouter()

# Global service instances (will be initialized in main.py)
question_service: QuestionService = None
grading_service: GradingService = None
openai_client = None


def get_question_service() -> QuestionService:
    """Dependency to get question service instance"""
    if question_service is None:
        raise HTTPException(status_code=500, detail="Question service not initialized")
    return question_service


def get_grading_service() -> GradingService:
    """Dependency to get grading service instance"""
    if grading_service is None:
        raise HTTPException(status_code=500, detail="Grading service not initialized")
    return grading_service


def _get_questions_directory_path(q_service: QuestionService) -> str:
    """
    Resolve the questions directory path using the same logic as QuestionService.
    """
    directory_path = settings.questions.default_file_path
    # Reuse the internal resolver for consistency
    return q_service._resolve_questions_directory_path(directory_path)  # type: ignore[attr-defined]


@router.get("/", response_model=HealthCheckResponse)
async def health_check(
    q_service: QuestionService = Depends(get_question_service)
) -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        Health status and number of questions loaded
    """
    return HealthCheckResponse(
        message="Answer Evaluator API is running",
        questions_loaded=q_service.get_questions_count()
    )


@router.get("/question", response_model=QuestionResponse)
async def get_random_question(
    q_service: QuestionService = Depends(get_question_service)
) -> QuestionResponse:
    """
    Get a random question from the question bank
    
    Returns:
        QuestionResponse with question_id and question_text
        
    Raises:
        HTTPException: If no questions are loaded
    """
    try:
        question = q_service.get_random_question()
        
        return QuestionResponse(
            question_id=question["question_id"],
            question_text=question["question_text"]
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=AnswerResponse)
async def grade_answer_endpoint(
    request: AnswerRequest,
    grading_service: GradingService = Depends(get_grading_service)
) -> AnswerResponse:
    """
    Grade a user's answer against the question's key points
    
    This endpoint:
    1. Validates the answer format and content
    2. Gets embedding for user answer
    3. Compares with precomputed key point embeddings
    4. Returns grade with score and feedback
    
    Args:
        request: AnswerRequest with question_id and user_answer
        
    Returns:
        AnswerResponse with score, hit/missing points, and feedback
        
    Raises:
        HTTPException: If question not found or grading fails
    """
    return grading_service.grade_answer(request.question_id, request.user_answer)


@router.get("/questions", response_model=AllQuestionsResponse)
async def get_all_questions(
    q_service: QuestionService = Depends(get_question_service)
) -> AllQuestionsResponse:
    """
    Get all questions (for debugging/development)
    
    Returns:
        AllQuestionsResponse with list of all questions
    """
    questions = q_service.get_all_questions()
    
    return AllQuestionsResponse(questions=questions)


@router.post("/question", response_model=AddQuestionResponse)
async def add_question(
    request: AddQuestionRequest,
    q_service: QuestionService = Depends(get_question_service),
    g_service: GradingService = Depends(get_grading_service)
) -> AddQuestionResponse:
    """
    Add a new question with key points to the database and vector store
    
    This endpoint:
    1. Adds the question to the questions bank
    2. Computes embeddings for all key points
    3. Stores embeddings in Pinecone vector database
    
    Args:
        request: AddQuestionRequest with question_text and key_points
        
    Returns:
        AddQuestionResponse with question_id and confirmation message
        
    Raises:
        HTTPException: If question is invalid or embedding computation fails
    """
    try:
        # Validate request
        if not request.question_text or not request.question_text.strip():
            raise HTTPException(status_code=400, detail="Question text cannot be empty")
        
        if not request.key_points or len(request.key_points) == 0:
            raise HTTPException(status_code=400, detail="At least one key point is required")
        
        # Convert key points to dictionary format
        key_points_dict = [
            {"text": kp.text, "weight": kp.weight}
            for kp in request.key_points
        ]
        
        # Add question to questions bank
        question_id = q_service.add_question(
            question_text=request.question_text.strip(),
            key_points=key_points_dict
        )
        
        # Get the newly added question
        new_question = q_service.get_question_by_id(question_id)
        
        if not new_question:
            raise HTTPException(status_code=500, detail="Failed to retrieve newly added question")
        
        # Compute and store embeddings
        print(f"🔄 Computing embeddings for new question {question_id}...")
        success = g_service.compute_and_store_embeddings_for_question(new_question)
        
        if not success:
            print(f"⚠️ Warning: Failed to store embeddings in Pinecone for question {question_id}")
            # Question is still added to the bank, embeddings will be computed on-demand
        
        return AddQuestionResponse(
            question_id=question_id,
            question_text=request.question_text.strip(),
            message=f"Question added successfully with {len(key_points_dict)} key point(s)"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add question: {str(e)}")


@router.delete("/question/{question_id}", response_model=DeleteQuestionResponse)
async def delete_question(
    question_id: int,
    q_service: QuestionService = Depends(get_question_service),
    g_service: GradingService = Depends(get_grading_service)
) -> DeleteQuestionResponse:
    """
    Delete a question from the database and remove its embeddings from the vector store
    
    This endpoint:
    1. Removes the question from the questions bank
    2. Removes embeddings from memory
    3. Deletes embeddings from Pinecone vector database
    
    Args:
        question_id: ID of the question to delete
        
    Returns:
        DeleteQuestionResponse with confirmation message
        
    Raises:
        HTTPException: If question not found or deletion fails
    """
    try:
        # Check if question exists
        if not q_service.question_exists(question_id):
            raise HTTPException(status_code=404, detail=f"Question with ID {question_id} not found")
        
        # Remove question from questions bank
        removed = q_service.remove_question(question_id)
        
        if not removed:
            raise HTTPException(status_code=500, detail=f"Failed to remove question {question_id} from questions bank")
        
        # Remove embeddings from memory and Pinecone
        print(f"🔄 Removing embeddings for question {question_id}...")
        embedding_removed = g_service.remove_question_embeddings(question_id)
        
        if not embedding_removed:
            print(f"⚠️ Warning: Failed to remove embeddings from Pinecone for question {question_id}")
            # Question is still removed from bank, but embeddings might remain in Pinecone
        
        return DeleteQuestionResponse(
            question_id=question_id,
            message=f"Question {question_id} deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete question: {str(e)}")


@router.post("/questions/bulk_from_text", response_model=BulkQuestionsFromTextResponse)
async def bulk_add_questions_from_text(
    request: BulkQuestionsFromTextRequest,
    q_service: QuestionService = Depends(get_question_service),
    g_service: GradingService = Depends(get_grading_service),
) -> BulkQuestionsFromTextResponse:
    """
    Generate multiple questions from a long teacher text using the LLM, transform them
    into rubric-style JSON (with key points), append them to rubric_added.json, and
    register them in the in-memory questions bank.
    """
    if not request.source_text or not request.source_text.strip():
        raise HTTPException(status_code=400, detail="Source text cannot be empty")

    if openai_client is None:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    source_text = request.source_text.strip()

    # 1) First LLM call – extract question texts from the long input
    try:
        questions_prompt = (
            "You are a teacher assistant that designs exam questions.\n"
            "From the following teaching text, infer the important concepts and generate a list\n"
            "of clear, distinct exam questions that assess understanding of those concepts.\n\n"
            "IMPORTANT:\n"
            "- Each question MUST be fully self-contained and make sense on its own.\n"
            "- DO NOT include phrases like \"according to the text\", \"in the passage\",\n"
            "  \"in the article\", \"in the story\", or similar references to a specific text.\n"
            "- Instead, phrase each question as a general question about the topic, as if it were\n"
            "  taken from a standalone exam (e.g. \"Explain how photosynthesis works in plants.\").\n"
            "- Avoid questions that only ask for decontextualized details like\n"
            "  \"What is the name of the city mentioned in the text?\" or similar.\n\n"
            "Return a JSON object with a single field 'questions', which is an array of strings.\n"
            "Each string should be one question.\n\n"
            f"Teaching text:\n{source_text}"
        )

        questions_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You generate exam questions and respond strictly in JSON.",
                },
                {"role": "user", "content": questions_prompt},
            ],
            max_tokens=800,
            temperature=0.3,
        )
        questions_content = questions_response.choices[0].message.content or ""
        questions_data = json.loads(questions_content)
        raw_questions = questions_data.get("questions", [])
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("LLM did not return a non-empty 'questions' list")

        question_texts = [str(q).strip() for q in raw_questions if str(q).strip()]
        if not question_texts:
            raise ValueError("No valid questions extracted from LLM response")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate questions from text: {e}",
        )

    # 2) Second LLM call – transform questions into rubric-style JSON with key points
    try:
        rubric_prompt = (
            "You are a teacher building a grading rubric.\n"
            "Given a list of exam questions, generate key points for each question.\n"
            "For every question, create 3–6 key points.\n"
            "Each key point should be a short, specific chunk (5–20 words) "
            "describing information a good answer should contain.\n"
            "Ensure the key points also include enough context so they can be understood\n"
            "without reading the original teaching text.\n\n"
            "Return a JSON object with a single field 'questions', which is an array.\n"
            "Each element must have:\n"
            "  - 'question_text': string (already self-contained)\n"
            "  - 'key_points': array of objects, each with:\n"
            "        'text': string\n"
            "        'weight': integer (usually 1)\n\n"
            f"Questions list:\n{json.dumps(question_texts, ensure_ascii=False)}"
        )

        rubric_response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You generate grading rubrics and respond strictly in JSON.",
                },
                {"role": "user", "content": rubric_prompt},
            ],
            max_tokens=2000,
            temperature=0.4,
        )
        rubric_content = rubric_response.choices[0].message.content or ""
        rubric_data = json.loads(rubric_content)
        rubric_questions = rubric_data.get("questions", [])
        if not isinstance(rubric_questions, list) or not rubric_questions:
            raise ValueError("LLM did not return a non-empty 'questions' list with rubrics")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate rubric-style questions: {e}",
        )

    # 3) Append to rubric_added.json and register in question service
    questions_dir = _get_questions_directory_path(q_service)
    os.makedirs(questions_dir, exist_ok=True)
    rubric_file = os.path.join(questions_dir, "rubric_added.json")

    existing: list[dict] = []
    if os.path.exists(rubric_file):
        try:
            with open(rubric_file, "r", encoding="utf-8") as f:
                existing = json.load(f) or []
        except Exception:
            existing = []

    added_questions: list[Question] = []

    for rq in rubric_questions:
        q_text = str(rq.get("question_text", "")).strip()
        kp_list = rq.get("key_points", []) or []
        if not q_text or not isinstance(kp_list, list) or not kp_list:
            continue

        # Normalize key points structure
        normalized_kps = []
        for kp in kp_list:
            kp_text = str(kp.get("text", "")).strip()
            if not kp_text:
                continue
            weight = kp.get("weight", 1)
            try:
                weight_int = int(weight)
            except Exception:
                weight_int = 1
            normalized_kps.append({"text": kp_text, "weight": weight_int})

        if not normalized_kps:
            continue

        # Generate a new question ID and register in memory with a dedicated category
        if not q_service._is_loaded:  # type: ignore[attr-defined]
            q_service.load_questions_bank()
        if q_service._questions_bank:  # type: ignore[attr-defined]
            max_id = max(q["question_id"] for q in q_service._questions_bank)  # type: ignore[attr-defined]
            new_id = max_id + 1
        else:
            new_id = 1

        question_dict = {
            "question_id": new_id,
            "question_text": q_text,
            "key_points": normalized_kps,
            "category": "rubric_added",
        }

        # Update in-memory structures
        q_service._questions_bank.append(question_dict)  # type: ignore[attr-defined]
        q_service._questions_by_id[new_id] = question_dict  # type: ignore[attr-defined]

        existing.append(
            {
                "question_id": new_id,
                "question_text": q_text,
                "key_points": normalized_kps,
                "category": "rubric_added",
            }
        )

        added_questions.append(
            Question(
                question_id=new_id,
                question_text=q_text,
                key_points=[  # type: ignore[list-item]
                    # Re-use KeyPoint-like dicts; Pydantic will coerce
                    {"text": kp["text"], "weight": kp["weight"]} for kp in normalized_kps
                ],
                category="rubric_added",
            )
        )

    # Persist rubric_added.json
    try:
        with open(rubric_file, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Questions generated but failed to save rubric_added.json: {e}",
        )

    # Optionally compute embeddings for the new questions (on-demand later or now)
    # For now, rely on on-demand computation in GradingService when a question is used.

    if not added_questions:
        raise HTTPException(
            status_code=500,
            detail="LLM did not produce any valid questions with key points",
        )

    return BulkQuestionsFromTextResponse(
        questions=added_questions,
        message=f"Successfully generated and added {len(added_questions)} question(s) from source text.",
    )
