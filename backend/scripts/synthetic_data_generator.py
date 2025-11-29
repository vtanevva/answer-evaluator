#!/usr/bin/env python3
"""
Synthetic Data Generator for Educational App
Generates biology questions with student answers and detailed scoring
"""

import os
import json
import csv
import time
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration - Easy to modify
M_QUESTIONS = 50  # Number of different questions to generate
N_ANSWERS_PER_QUESTION = 20  # Number of student answers per question
OUTPUT_FORMAT = "jsonl"  # "jsonl" or "csv"
OUTPUT_FILE = "synthetic_biology_data.jsonl"

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_ideal_points(client: OpenAI) -> Dict[str, Any]:
    """
    Generate a biology question and its ideal answer points using GPT-4o-mini
    """
    prompt = """
    Create a biology question suitable for middle school students (ages 12-14) and provide 3-5 ideal answer points.
    
    Return your response as a JSON object with this exact structure:
    {
        "question": "Your biology question here",
        "ideal_answer_points": [
            "First key point",
            "Second key point", 
            "Third key point"
        ]
    }
    
    Requirements:
    - Question should be about biology (animals, plants, ecosystems, human body, etc.)
    - Each ideal point should be a complete, standalone statement
    - Keep points concise but informative (1-2 sentences max)
    - Make sure points are distinct and don't overlap significantly
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        print(f"Error generating ideal points: {e}")
        # Fallback question if API fails
        return {
            "question": "What are three adaptations that help desert animals survive?",
            "ideal_answer_points": [
                "Desert animals can store water in their bodies",
                "Many desert animals are active at night to avoid heat",
                "Some desert animals have light-colored skin to reflect sunlight"
            ]
        }

def generate_student_answers(client: OpenAI, question: str, ideal_points: List[str]) -> List[Dict[str, Any]]:
    """
    Generate N student answers for a given question, with varying correctness levels
    """
    prompt = f"""
    Given this biology question and ideal answer points, generate {N_ANSWERS_PER_QUESTION} different student answers.
    
    Question: "{question}"
    Ideal answer points: {ideal_points}
    
    Create a variety of student answers:
    - Some completely correct (mentioning all points)
    - Some partially correct (mentioning 1-2 points)
    - Some incorrect or irrelevant
    - Vary the writing style and length
    
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
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=2000
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
                "score": 66.7,
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

def main():
    """Main function to generate synthetic data"""
    print(f"🚀 Starting synthetic data generation...")
    print(f"📊 Configuration:")
    print(f"   - Questions to generate: {M_QUESTIONS}")
    print(f"   - Answers per question: {N_ANSWERS_PER_QUESTION}")
    print(f"   - Total records: {M_QUESTIONS * N_ANSWERS_PER_QUESTION:,}")
    print(f"   - Output format: {OUTPUT_FORMAT}")
    print(f"   - Output file: {OUTPUT_FILE}")
    print("-" * 50)
    
    all_records = []
    start_time = time.time()
    
    for question_num in range(1, M_QUESTIONS + 1):
        print(f"📝 Generating question {question_num}/{M_QUESTIONS}...")
        
        # Generate question and ideal points
        question_data = generate_ideal_points(client)
        question = question_data["question"]
        ideal_points = question_data["ideal_answer_points"]
        
        print(f"   Question: {question[:60]}{'...' if len(question) > 60 else ''}")
        print(f"   Ideal points: {len(ideal_points)}")
        
        # Generate student answers
        student_answers = generate_student_answers(client, question, ideal_points)
        
        # Create final records
        question_records = []
        for student_data in student_answers:
            record = create_final_record(question, ideal_points, student_data)
            question_records.append(record)
        
        all_records.extend(question_records)
        
        print(f"   ✅ Generated {len(question_records)} answers")
        print(f"   📈 Total records so far: {len(all_records):,}")
        
        # Add small delay to be respectful to API
        time.sleep(0.5)
        print()
    
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
    print(f"   - Questions generated: {M_QUESTIONS}")
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
