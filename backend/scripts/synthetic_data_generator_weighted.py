#!/usr/bin/env python3
"""
Weighted Synthetic Data Generator for Educational App
- Numeric question IDs
- Weighted key points format
- New JSON structure
"""

import os
import json
import csv
import time
import random
from typing import List, Dict, Any
from dotenv import load_dotenv
from config_weighted import *

# Load environment variables from .env file
load_dotenv()

# Import OpenAI with backward compatibility
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    OPENAI_NEW_VERSION = True
except ImportError:
    import openai
    openai.api_key = os.environ["OPENAI_API_KEY"]
    OPENAI_NEW_VERSION = False

# Global token tracking
total_input_tokens = 0
total_output_tokens = 0
seen_questions: set[str] = set()

def update_token_tracking(input_tokens: int, output_tokens: int):
    """Update global token tracking"""
    global total_input_tokens, total_output_tokens

    total_input_tokens += input_tokens
    total_output_tokens += output_tokens

def _parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from model output; tolerate code fences and extra text."""
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.strip('`')
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception:
            pass
    raise ValueError("Invalid JSON from model")


def generate_weighted_question(question_id: int, prior_examples: List[str] | None = None) -> Dict[str, Any]:
    """
    Generate a question with weighted key points using GPT-4o-mini.
    Returns a dictionary describing the question.
    """
    # Randomly select a question type
    question_type = random.choice(QUESTION_TYPES)
    
    prior_hint = "\nPreviously generated questions (avoid repeating phrasing):\n- " + "\n- ".join(prior_examples[:10]) if prior_examples else ""
    prompt = f"""
    Create a {SUBJECT} question suitable for {GRADE_LEVEL} students ({STUDENT_AGE_RANGE}) about {question_type} and provide 2-4 key points with weights.
    
    Return your response as a JSON object with this exact structure:
    {{
        "question_text": "Your {SUBJECT} question here",
        "key_points": [
            {{"text": "First key point", "weight": 1}},
            {{"text": "Second key point", "weight": 1}},
            {{"text": "Third key point", "weight": 1}}
        ]
    }}
    
    Requirements:
    - Question should be about {SUBJECT} with focus on {question_type}
    - Each key point should be SHORT, SPECIFIC chunks (5-15 words max)
    - Make key points DISTINCT and NON-OVERLAPPING for easy grading
    - Each key point should be a separate, identifiable concept
    - All weights should be 1 (equal importance)
    - Ensure the question is appropriate for {GRADE_LEVEL} level
    - MUST return exactly 2-4 key points
    - Points should be granular enough to track partial credit
    - Do NOT repeat earlier question texts/phrasing in this session; vary subtopics and wording.
    
    Examples of good key points:
    - "General increase in prices"
    - "Reduction of purchasing power"
    - "Decrease in money value"
    {prior_hint}
    """
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            
            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            update_token_tracking(input_tokens, output_tokens)

            result = _parse_json_response(response.choices[0].message.content)
        else:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            
            # Extract token usage (older API format)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            update_token_tracking(input_tokens, output_tokens)

            result = _parse_json_response(response.choices[0].message.content)
        
        # Add question_id to the result
        result["question_id"] = question_id
        
        return result
        
    except Exception as e:
        print(f"Error generating weighted question: {e}")
        # Fallback questions with weighted key points
        fallback_questions = [
            {
                "question_id": question_id,
                "question_text": "What are three adaptations that help desert animals survive?",
                "key_points": [
                    {"text": "Store water in their bodies", "weight": 1},
                    {"text": "Active at night", "weight": 1},
                    {"text": "Light-colored skin", "weight": 1}
                ]
            },
            {
                "question_id": question_id,
                "question_text": "How do plants make their own food?",
                "key_points": [
                    {"text": "Use sunlight as energy", "weight": 1},
                    {"text": "Take in carbon dioxide", "weight": 1},
                    {"text": "Absorb water through roots", "weight": 1}
                ]
            },
            {
                "question_id": question_id,
                "question_text": "What are the main functions of the circulatory system?",
                "key_points": [
                    {"text": "Transports oxygen to cells", "weight": 1},
                    {"text": "Carries nutrients to body", "weight": 1},
                    {"text": "Removes waste products", "weight": 1}
                ]
            },
            {
                "question_id": question_id,
                "question_text": "Explain inflation",
                "key_points": [
                    {"text": "General increase in prices", "weight": 1},
                    {"text": "Reduction of purchasing power", "weight": 1}
                ]
            }
        ]
        return random.choice(fallback_questions)

def save_weighted_data(records: List[Dict[str, Any]], filename: str):
    """Save weighted records to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def print_progress_weighted(question_num: int, total_questions: int):
    """Print progress information"""
    if SHOW_PROGRESS:
        percentage = (question_num / total_questions) * 100
        print(f"📊 Progress: {question_num}/{total_questions} questions ({percentage:.1f}%)")

def main():
    """Main function to generate weighted synthetic data"""
    print(f"🚀 Starting WEIGHTED synthetic data generation...")
    print(f"📊 Configuration:")
    print(f"   - Subject: {SUBJECT}")
    print(f"   - Grade level: {GRADE_LEVEL}")
    print(f"   - Questions to generate: {M_QUESTIONS}")
    print(f"   - Output format: JSON array")
    print(f"   - Output file: weighted_biology_questions.json")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Question types: {', '.join(QUESTION_TYPES)}")
    print("-" * 60)
    
    all_records = []
    start_time = time.time()
    
    try:
        for question_num in range(1, M_QUESTIONS + 1):
            if SHOW_PROGRESS:
                print(f"📝 Generating weighted question {question_num}/{M_QUESTIONS}...")
            
            # Generate with retries and de-duplication
            question_data = None
            for attempt in range(4):
                candidate = generate_weighted_question(question_num, prior_examples=list(seen_questions))
                qt = candidate.get("question_text", "").strip().lower()
                if qt and qt not in seen_questions:
                    question_data = candidate
                    seen_questions.add(qt)
                    break
                time.sleep(0.2)
            if question_data is None:
                candidate = generate_weighted_question(question_num)
                qt = candidate.get("question_text", "Untitled question")
                candidate["question_text"] = f"{qt} (variant {question_num})"
                question_data = candidate
                seen_questions.add(question_data["question_text"].lower())
            
            if SHOW_PROGRESS:
                print(f"   Question: {question_data['question_text'][:60]}{'...' if len(question_data['question_text']) > 60 else ''}")
                print(f"   Key points: {len(question_data['key_points'])}")
            
            all_records.append(question_data)
            
            if SHOW_PROGRESS:
                print_progress_weighted(question_num, M_QUESTIONS)
            
            # Add delay to be respectful to API
            time.sleep(API_DELAY)
            print()
    
    except KeyboardInterrupt:
        print(f"\nGeneration interrupted by user. Saving {len(all_records)} questions generated so far...")
    
    # Save results
    output_file = "generated_data/economy.json"
    print(f"💾 Saving {len(all_records)} questions to {output_file}...")
    save_weighted_data(all_records, output_file)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Statistics:")
    print(f"   - Total questions: {len(all_records)}")
    print(f"   - Time taken: {duration:.1f} seconds")
    questions_per_second = (len(all_records) / duration) if duration > 0 else 0.0
    print(f"   - Questions per second: {questions_per_second:.1f}")
    print(f"   - Output file: {output_file}")
    print(f"Token usage:")
    print(f"   - Total input tokens: {total_input_tokens:,}")
    print(f"   - Total output tokens: {total_output_tokens:,}")
    print(f"   - Total tokens: {total_input_tokens + total_output_tokens:,}")
    
    # Show sample record
    if all_records:
        print(f"\n📄 Sample record:")
        print(json.dumps(all_records[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        exit(1)
    
    main()
