#!/usr/bin/env python3
"""
Enhanced Synthetic Data Generator for Educational App
Uses configuration file for easy customization
"""

import os
import json
import csv
import time
import random
from typing import List, Dict, Any
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

def generate_ideal_points() -> Dict[str, Any]:
    """
    Generate a question and its ideal answer points using GPT-4o-mini
    """
    # Randomly select a question type
    question_type = random.choice(QUESTION_TYPES)
    
    prompt = f"""
    Create a {SUBJECT} question suitable for {GRADE_LEVEL} students ({STUDENT_AGE_RANGE}) about {question_type} and provide 3-5 ideal answer points.
    
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
    - Each ideal point should be a complete, standalone statement
    - Keep points concise but informative (1-2 sentences max)
    - Make sure points are distinct and don't overlap significantly
    - Ensure the question is appropriate for {GRADE_LEVEL} level
    """
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            result = json.loads(response.choices[0].message.content.strip())
        else:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_QUESTION,
                max_tokens=MAX_TOKENS_QUESTION
            )
            result = json.loads(response.choices[0].message.content.strip())
        
        return result
        
    except Exception as e:
        print(f"Error generating ideal points: {e}")
        # Fallback questions by type
        fallback_questions = {
            "animals": {
                "question": "What are three adaptations that help desert animals survive?",
                "ideal_answer_points": [
                    "Desert animals can store water in their bodies",
                    "Many desert animals are active at night to avoid heat",
                    "Some desert animals have light-colored skin to reflect sunlight"
                ]
            },
            "plants": {
                "question": "How do plants make their own food?",
                "ideal_answer_points": [
                    "Plants use sunlight as an energy source",
                    "Plants take in carbon dioxide from the air",
                    "Plants absorb water through their roots",
                    "The process is called photosynthesis"
                ]
            },
            "human body": {
                "question": "What are the main functions of the circulatory system?",
                "ideal_answer_points": [
                    "It transports oxygen to body cells",
                    "It carries nutrients throughout the body",
                    "It removes waste products from cells",
                    "It helps regulate body temperature"
                ]
            }
        }
        return fallback_questions.get(question_type, fallback_questions["animals"])

def generate_student_answers(question: str, ideal_points: List[str]) -> List[Dict[str, Any]]:
    """
    Generate N student answers for a given question, with varying correctness levels
    """
    prompt = f"""
    Given this {SUBJECT} question and ideal answer points, generate {N_ANSWERS_PER_QUESTION} different student answers.
    
    Question: "{question}"
    Ideal answer points: {ideal_points}
    
    Create a variety of student answers:
    - Some completely correct (mentioning all points)
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
    """
    
    try:
        if OPENAI_NEW_VERSION:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_ANSWERS,
                max_tokens=MAX_TOKENS_ANSWERS
            )
            result = json.loads(response.choices[0].message.content.strip())
        else:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE_ANSWERS,
                max_tokens=MAX_TOKENS_ANSWERS
            )
            result = json.loads(response.choices[0].message.content.strip())
        
        return result
        
    except Exception as e:
        print(f"Error generating student answers: {e}")
        # Fallback answers if API fails
        return [
            {
                "student_answer": "Desert animals can store water and are active at night.",
                "covered_points": ideal_points[:2],
                "missing_points": ideal_points[2:],
                "score": round((2 / len(ideal_points)) * 100, 1),
                "feedback": "Good start! You mentioned water storage and nocturnal activity. You missed mentioning light-colored skin for heat reflection."
            }
        ]

def create_final_record(question: str, ideal_points: List[str], student_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine question data with student answer data to create final record
    """
    return {
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

def print_progress(question_num: int, total_questions: int, records_generated: int):
    """Print progress information"""
    if SHOW_PROGRESS:
        percentage = (question_num / total_questions) * 100
        print(f"📊 Progress: {question_num}/{total_questions} questions ({percentage:.1f}%) - {records_generated:,} records")

def main():
    """Main function to generate synthetic data"""
    print(f"🚀 Starting enhanced synthetic data generation...")
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
            question_data = generate_ideal_points()
            question = question_data["question"]
            ideal_points = question_data["ideal_answer_points"]
            
            if SHOW_PROGRESS:
                print(f"   Question: {question[:60]}{'...' if len(question) > 60 else ''}")
                print(f"   Ideal points: {len(ideal_points)}")
            
            # Generate student answers
            student_answers = generate_student_answers(question, ideal_points)
            
            # Create final records
            question_records = []
            for student_data in student_answers:
                record = create_final_record(question, ideal_points, student_data)
                question_records.append(record)
            
            all_records.extend(question_records)
            
            if SHOW_PROGRESS:
                print(f"   ✅ Generated {len(question_records)} answers")
                print_progress(question_num, M_QUESTIONS, len(all_records))
            
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
