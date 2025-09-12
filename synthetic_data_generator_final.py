#!/usr/bin/env python3
"""
Final Synthetic Data Generator for Educational App
- Exact JSON schema format
- Token usage tracking with cost calculation
- 100 samples per run
- Uses gpt-4o-mini
"""

import os
import json
import csv
import time
import random
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from config import *

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
total_cost_usd = 0.0

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate cost based on token usage"""
    input_cost = (input_tokens / 1000) * GPT4O_MINI_INPUT_COST_PER_1K_TOKENS
    output_cost = (output_tokens / 1000) * GPT4O_MINI_OUTPUT_COST_PER_1K_TOKENS
    return input_cost + output_cost

def update_token_tracking(input_tokens: int, output_tokens: int):
    """Update global token tracking"""
    global total_input_tokens, total_output_tokens, total_cost_usd
    
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    cost = calculate_cost(input_tokens, output_tokens)
    total_cost_usd += cost
    
    return cost

def generate_ideal_points() -> Tuple[Dict[str, Any], float]:
    """
    Generate a question and its ideal answer points using GPT-4o-mini
    Returns: (question_data, cost_usd)
    """
    # Randomly select a question type
    question_type = random.choice(QUESTION_TYPES)
    
    prompt = f"""
    Create a {SUBJECT} question suitable for {GRADE_LEVEL} students ({STUDENT_AGE_RANGE}) about {question_type} and provide exactly 3 ideal answer points.
    
    Return your response as a JSON object with this exact structure:
    {{
        "question": "Your {SUBJECT} question here",
        "ideal_answer_points": [
            "First key point",
            "Second key point", 
            "Third key point"
        ]
    }}
    
    Requirements:
    - Question should be about {SUBJECT} with focus on {question_type}
    - Each ideal point should be a SHORT, SPECIFIC chunk that can be easily detected in student answers
    - Keep points concise (1 short sentence or phrase, 5-15 words max)
    - Make points DISTINCT and NON-OVERLAPPING for easy grading
    - Each point should be a separate, identifiable concept
    - Ensure the question is appropriate for {GRADE_LEVEL} level
    - MUST return exactly 3 ideal answer points
    - Points should be granular enough to track partial credit
    
    Examples of good ideal points:
    - "Butterflies have colorful wings"
    - "Butterflies can see ultraviolet light"
    - "Butterflies use pheromones to communicate"
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
            cost = update_token_tracking(input_tokens, output_tokens)
            
            result = json.loads(response.choices[0].message.content.strip())
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
            cost = update_token_tracking(input_tokens, output_tokens)
            
            result = json.loads(response.choices[0].message.content.strip())
        
        return result, cost
        
    except Exception as e:
        print(f"Error generating ideal points: {e}")
        # Fallback questions by type with chunked ideal points
        fallback_questions = {
            "animals": {
                "question": "What are three adaptations that help desert animals survive?",
                "ideal_answer_points": [
                    "Store water in their bodies",
                    "Active at night",
                    "Light-colored skin"
                ]
            },
            "plants": {
                "question": "How do plants make their own food?",
                "ideal_answer_points": [
                    "Use sunlight as energy",
                    "Take in carbon dioxide",
                    "Absorb water through roots"
                ]
            },
            "human body": {
                "question": "What are the main functions of the circulatory system?",
                "ideal_answer_points": [
                    "Transports oxygen to cells",
                    "Carries nutrients to body",
                    "Removes waste products"
                ]
            },
            "butterflies": {
                "question": "Describe three characteristics of butterflies.",
                "ideal_answer_points": [
                    "Have colorful wings",
                    "Can see ultraviolet light", 
                    "Use pheromones to communicate"
                ]
            }
        }
        return fallback_questions.get(question_type, fallback_questions["animals"]), 0.0

def generate_student_answers(question: str, ideal_points: List[str]) -> Tuple[List[Dict[str, Any]], float]:
    """
    Generate N student answers for a given question, with varying correctness levels
    Returns: (student_answers, cost_usd)
    """
    prompt = f"""
    Given this {SUBJECT} question and ideal answer points, generate {N_ANSWERS_PER_QUESTION} different student answers.
    
    Question: "{question}"
    Ideal answer points: {ideal_points}
    
    Create a variety of student answers:
    - Some completely correct (mentioning all 3 points)
    - Some partially correct (mentioning 1-2 points) 
    - Some incorrect or irrelevant
    - Vary the writing style and length to simulate real students
    - Include some answers with minor misconceptions
    - Make some answers very brief, others more detailed
    
    For each student answer, return a JSON object with this exact structure:
    {{
        "student_answer": "The student's response text",
        "covered_points": ["List of ideal points the student mentioned"],
        "missing_points": ["List of ideal points the student missed"],
        "score": 85.5,
        "feedback": "Encouraging feedback with specific suggestions"
    }}
    
    Return all {N_ANSWERS_PER_QUESTION} answers as a JSON array.
    
    Requirements:
    - Calculate score as (covered_points / total_points) * 100
    - Write encouraging but constructive feedback
    - Make student answers sound natural (varying grammar, length, style)
    - Ensure covered_points and missing_points together equal all ideal_points
    - Include feedback that helps students improve
    - Score should be rounded to 1 decimal place
    - Match student answers against the SPECIFIC ideal points provided
    - Only mark a point as "covered" if the student's answer contains the key concept from that ideal point
    """
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_ANSWERS,
                max_tokens=MAX_TOKENS_ANSWERS
            )
            
            # Extract token usage
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = update_token_tracking(input_tokens, output_tokens)
            
            result = json.loads(response.choices[0].message.content.strip())
        else:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_ANSWERS,
                max_tokens=MAX_TOKENS_ANSWERS
            )
            
            # Extract token usage (older API format)
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = update_token_tracking(input_tokens, output_tokens)
            
            result = json.loads(response.choices[0].message.content.strip())
        
        return result, cost
        
    except Exception as e:
        print(f"Error generating student answers: {e}")
        # Fallback answers if API fails
        fallback_answers = []
        for i in range(min(5, N_ANSWERS_PER_QUESTION)):  # Generate 5 fallback answers
            covered_count = random.randint(0, len(ideal_points))
            covered_points = ideal_points[:covered_count]
            missing_points = ideal_points[covered_count:]
            score = round((covered_count / len(ideal_points)) * 100, 1) if ideal_points else 0.0
            
            # Create more realistic student answers
            if covered_count == 0:
                student_answer = "I don't know the answer to this question."
            elif covered_count == 1:
                student_answer = f"Based on what I know, {covered_points[0].lower()}."
            elif covered_count == 2:
                student_answer = f"I think {covered_points[0].lower()} and also {covered_points[1].lower()}."
            else:
                student_answer = f"{covered_points[0].lower()}, {covered_points[1].lower()}, and {covered_points[2].lower()}."
            
            fallback_answers.append({
                "student_answer": student_answer,
                "covered_points": covered_points,
                "missing_points": missing_points,
                "score": score,
                "feedback": f"Good work! You mentioned {len(covered_points)} out of {len(ideal_points)} key points. Keep studying!" if covered_count > 0 else "Don't give up! Review the material and try again."
            })
        
        return fallback_answers, 0.0

def create_final_record(question: str, ideal_points: List[str], student_data: Dict[str, Any], question_id: str, record_id: int) -> Dict[str, Any]:
    """
    Combine question data with student answer data to create final record
    Returns the exact schema format requested with question ID
    """
    return {
        "question_id": question_id,
        "question": question,
        "ideal_answer_points": ideal_points,
        "student_answer": student_data["student_answer"],
        "covered_points": student_data["covered_points"],
        "missing_points": student_data["missing_points"],
        "score": student_data["score"],
        "feedback": student_data["feedback"]
    }

def save_to_jsonl(records: List[Dict[str, Any]], filename: str):
    """Save records to JSONL file"""
    with open(filename, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def save_to_csv(records: List[Dict[str, Any]], filename: str):
    """Save records to CSV file"""
    if not records:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

def print_progress(question_num: int, total_questions: int, records_generated: int, current_cost: float):
    """Print progress information with cost tracking"""
    if SHOW_PROGRESS:
        percentage = (question_num / total_questions) * 100
        print(f"📊 Progress: {question_num}/{total_questions} questions ({percentage:.1f}%) - {records_generated:,} records - Cost: ${current_cost:.4f}")

def main():
    """Main function to generate synthetic data"""
    print(f"🚀 Starting FINAL synthetic data generation...")
    print(f"📊 Configuration:")
    print(f"   - Subject: {SUBJECT}")
    print(f"   - Grade level: {GRADE_LEVEL}")
    print(f"   - Questions to generate: {M_QUESTIONS}")
    print(f"   - Answers per question: {N_ANSWERS_PER_QUESTION}")
    print(f"   - Total records: {M_QUESTIONS * N_ANSWERS_PER_QUESTION:,}")
    print(f"   - Output format: {OUTPUT_FORMAT}")
    print(f"   - Output file: {OUTPUT_FILE}")
    print(f"   - Model: {MODEL_NAME}")
    print(f"   - Question types: {', '.join(QUESTION_TYPES)}")
    print("-" * 60)
    
    all_records = []
    start_time = time.time()
    
    try:
        for question_num in range(1, M_QUESTIONS + 1):
            if SHOW_PROGRESS:
                print(f"📝 Generating question {question_num}/{M_QUESTIONS}...")
            
            # Generate question and ideal points
            question_data, question_cost = generate_ideal_points()
            question = question_data["question"]
            ideal_points = question_data["ideal_answer_points"]
            
            if SHOW_PROGRESS:
                print(f"   Question: {question[:60]}{'...' if len(question) > 60 else ''}")
                print(f"   Ideal points: {len(ideal_points)}")
                print(f"   Question generation cost: ${question_cost:.4f}")
            
            # Generate student answers
            student_answers, answers_cost = generate_student_answers(question, ideal_points)
            
            if SHOW_PROGRESS:
                print(f"   Student answers generation cost: ${answers_cost:.4f}")
            
            # Create final records with question ID
            question_id = f"Q{question_num:03d}_{int(time.time())}"  # e.g., "Q001_1703123456"
            question_records = []
            
            for i, student_data in enumerate(student_answers):
                record_id = i + 1
                record = create_final_record(question, ideal_points, student_data, question_id, record_id)
                question_records.append(record)
                
                # Print progress after each sample
                if SHOW_PROGRESS and (i + 1) % 10 == 0:  # Every 10 samples
                    print(f"   ✅ Generated {i + 1}/{len(student_answers)} answers")
            
            all_records.extend(question_records)
            
            if SHOW_PROGRESS:
                print(f"   ✅ Generated {len(question_records)} total answers")
                print(f"   Question ID: {question_id}")
                print_progress(question_num, M_QUESTIONS, len(all_records), total_cost_usd)
            
            # Add delay to be respectful to API
            time.sleep(API_DELAY)
            print()
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Generation interrupted by user. Saving {len(all_records):,} records generated so far...")
    
    # Save results
    print(f"💾 Saving {len(all_records):,} records to {OUTPUT_FILE}...")
    
    if OUTPUT_FORMAT.lower() == "csv":
        save_to_csv(all_records, OUTPUT_FILE)
    else:
        save_to_jsonl(all_records, OUTPUT_FILE)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("✅ Generation complete!")
    print(f"📊 Final Statistics:")
    print(f"   - Total records: {len(all_records):,}")
    print(f"   - Questions generated: {len(set(record['question'] for record in all_records))}")
    print(f"   - Answers per question: {N_ANSWERS_PER_QUESTION}")
    print(f"   - Time taken: {duration:.1f} seconds")
    print(f"   - Records per second: {len(all_records)/duration:.1f}")
    print(f"   - Output file: {OUTPUT_FILE}")
    print(f"💰 Cost Analysis:")
    print(f"   - Total input tokens: {total_input_tokens:,}")
    print(f"   - Total output tokens: {total_output_tokens:,}")
    print(f"   - Total tokens: {total_input_tokens + total_output_tokens:,}")
    print(f"   - Total cost: ${total_cost_usd:.4f}")
    print(f"   - Cost per record: ${total_cost_usd/len(all_records):.4f}")
    
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
